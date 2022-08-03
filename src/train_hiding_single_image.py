import os
import sys
from argparse import Namespace
import time

# 导入pytorch包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger

from data_loader import ImageFolder, RandomBatchLoader
from utils import get_config, get_args, dump_cfg
from utils import save_imgs
from mytime import time_change
from encoded_processing import to_binary_string, to_tensor, XOR

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))
from cae_512 import CAE as CAE512
from refining_unet import PixDenoiseModel

HIDDEN_AMOUNT = 1


def prologue(cfg: Namespace, *varargs) -> None:
    # sanity checks
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))


def train(cfg: Namespace) -> None:
    logger.info("=== Training ===")

    prologue(cfg)

    # CAE Models
    cover_model = CAE512()
    secret_model = CAE512()

    cover_model.load_model(cfg.cover_encoder_ckpt, cfg.cover_decoder_ckpt)
    secret_model.load_model(cfg.secret_encoder_ckpt, cfg.secret_decoder_ckpt)

    cover_model.encoder.eval()
    secret_model.encoder.eval()

    cover_model.decoder.train()
    secret_model.decoder.train()

    # Denoise Models
    cover_denoise_model = PixDenoiseModel()
    secret_denoise_model = PixDenoiseModel()

    cover_denoise_model.load_model(cfg.cover_denoise_ckpt)
    secret_denoise_model.load_model(cfg.secret_denoise_ckpt)

    cover_denoise_model.train()
    secret_denoise_model.train()

    if cfg.device == "cuda":
        cover_model.cuda()
        secret_model.cuda()
        cover_denoise_model.cuda()
        secret_denoise_model.cuda()

    logger.debug(f"Model loaded on {cfg.device}")

    cover_dataset = ImageFolder(cfg.cover_dataset_path)
    cover_dataloader = DataLoader(cover_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers)
    secret_dataloader = RandomBatchLoader(cfg.secret_dataset_path)
    logger.debug("Data loaded")

    secret_decoder_optimizer = optim.Adam(secret_model.decoder.parameters(), lr=cfg.learning_rate)

    loss_criterion = nn.MSELoss()

    cover_loss_avg, denoise_cover_loss_avg = 0.0, 0.0
    secret_loss_avg, denoise_secret_loss_avg = 0.0, 0.0

    time_start = time.time()
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):
        for batch_idx, data in enumerate(cover_dataloader, start=1):
            cover, _ = data
            batch_size = cover.shape[0]
            secrets = []
            for i in range(HIDDEN_AMOUNT):
                secrets.append(secret_dataloader.fetch_data(batch_size)[0])
            secret = torch.cat(secrets, dim=0)

            if cfg.device == "cuda":
                cover = cover.cuda()
                secret = secret.cuda()

            secret_decoder_optimizer.zero_grad()

            cover = Variable(cover)
            secret = Variable(secret)

            cover_encoded = cover_model.encode(cover)
            secret_encoded = (secret_model.encode(secret))
            secret_encodeds = torch.chunk(secret_encoded, chunks=HIDDEN_AMOUNT, dim=0)
            secret_encoded = torch.cat(secret_encodeds, dim=1)

            key = XOR(to_binary_string(cover_encoded), to_binary_string(secret_encoded))

            cover_out = cover_model.decode(cover_encoded)
            cover_loss = loss_criterion(cover_out, cover)
            cover_loss_avg += cover_loss.item()

            cover_denoise_model.real_A = cover_out
            cover_denoise_model.real_B = cover
            cover_denoise_model.forward()
            cover_out_denoise = cover_denoise_model.fake_B
            denoise_cover_loss_avg += loss_criterion(cover_out_denoise, cover).item()

            # Recover
            cover_encoded = cover_model.encode(cover_out_denoise)
            recovered_secret_encoded = to_tensor(XOR(to_binary_string(cover_encoded), key))
            recovered_secret_encodeds = torch.chunk(recovered_secret_encoded, chunks=HIDDEN_AMOUNT, dim=1)
            recovered_secret_encoded = torch.cat(recovered_secret_encodeds, dim=0)

            recovered_secret = secret_model.decode(recovered_secret_encoded)

            secret_loss = loss_criterion(recovered_secret, secret)
            secret_loss_avg += secret_loss.item()

            secret_decoder_optimizer.zero_grad()
            secret_loss.backward()
            secret_decoder_optimizer.step()

            secret_denoise_model.real_A = recovered_secret.detach()
            secret_denoise_model.real_B = secret.detach()
            secret_denoise_model.optimize_parameters()
            denoised_recovered = secret_denoise_model.fake_B

            denoise_secret_loss_avg += loss_criterion(denoised_recovered, secret).item()

            # -- batch-loop

        if epoch_idx % cfg.print_every == 0:
            time_end = time.time()
            time_used = time_end - time_start
            logger.debug(
                '[%3d/%3d] \n'
                '          cover_loss: %.8f denoised_cover_loss: %.8f\n'
                '          secret_loss: %.8f    denoised_secret_loss:    %.8f\n'
                '          used time: %s rest_time: %s' %
                (epoch_idx, cfg.num_epochs,
                 cover_loss_avg / (cfg.print_every * len(cover_dataloader)),
                 denoise_cover_loss_avg / (cfg.print_every * len(cover_dataloader)),
                 secret_loss_avg / (cfg.print_every * len(cover_dataloader)),
                 denoise_secret_loss_avg / (cfg.print_every * len(cover_dataloader)),
                 time_change(time_used), time_change(
                    time_used / (epoch_idx - cfg.start_epoch + 1) * (cfg.num_epochs - epoch_idx))))
            cover_loss_avg, denoise_cover_loss_avg = 0.0, 0.0
            secret_loss_avg, denoise_secret_loss_avg = 0.0, 0.0

        if epoch_idx % cfg.save_img_every == 0:
            top = [cover[0].cpu().data]
            for s in secrets:
                top.append(s[0].cpu().data)
            top = torch.cat(top, dim=2)

            middle = [cover_out[0].cpu().data]
            recovered_secrets = torch.chunk(recovered_secret, chunks=HIDDEN_AMOUNT, dim=0)
            for r in recovered_secrets:
                middle.append(r[0].cpu().data)
            middle = torch.cat(middle, dim=2)

            bottom = [cover_out_denoise[0].cpu().data]
            denoised_recovereds = torch.chunk(denoised_recovered, chunks=HIDDEN_AMOUNT, dim=0)
            for d in denoised_recovereds:
                bottom.append(d[0].cpu().data)
            bottom = torch.cat(bottom, dim=2)

            out = torch.cat([top, middle, bottom], dim=1)

            save_imgs(imgs=out,
                      name="../experiments/%s/out/out_%05d.png" % (cfg.exp_name, epoch_idx))

        if epoch_idx % cfg.save_model_every == 0:
            secret_model.save_model(cfg.exp_name, 'secret_' + str(epoch_idx))
            secret_denoise_model.save_model(cfg.exp_name, 'secret_denoise_' + str(epoch_idx))

if __name__ == '__main__':
    args = get_args()
    config = get_config(args)
    train(config)
