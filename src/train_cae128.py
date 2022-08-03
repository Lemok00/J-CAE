import os
import sys
from argparse import Namespace
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder
from utils import get_config, get_args, dump_cfg
from utils import save_imgs
from mytime import time_change

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))
from cae_128 import CAE


def prologue(cfg: Namespace, *varargs) -> SummaryWriter:
    # sanity checks
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))

    # tb writer
    writer = SummaryWriter(f"{base_dir}/logs")

    return writer


def epilogue(cfg: Namespace, *varargs) -> None:
    writer = varargs[0]
    writer.close()


def train(cfg: Namespace) -> None:
    logger.info("=== Training ===")

    # initial setup
    writer = prologue(cfg)

    # train-related code
    model = CAE()
    model.train()
    if cfg.device == "cuda":
        model.cuda()
    logger.debug(f"Model loaded on {cfg.device}")

    # continue-train
    if (cfg._continue == True):
        model.load_model(cfg.encoder_ckpt, cfg.decoder_ckpt)

    # Datasets
    dataset = ImageFolder(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    logger.debug("Data loaded")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    # Loss -> MSE
    loss_criterion = nn.MSELoss()
    loss_encoder = nn.MSELoss()
    # scheduler = ...

    epoch_avg = 0.0
    encoder_avg = 0.0
    ts = 0

    # train-loop
    # epoch
    time_start = time.time()
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):

        # scheduler.step()
        # batch-loop
        for batch_idx, data in enumerate(dataloader, start=1):
            img, _ = data
            # [0,1]->[-1,1]
            img = img * 2 - 1

            if cfg.device == "cuda":
                patch = img.cuda()

            optimizer.zero_grad()
            optimizer_encoder.zero_grad()

            x = Variable(patch)
            y = model(x)
            loss = loss_criterion(y, x)

            avg_loss_per_image = loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_avg += avg_loss_per_image

            x_encoded = model.encoder(x)
            y_encoded = model.encoder(y)
            encoder_loss = loss_encoder(y_encoded, x_encoded)

            encoder_loss_per_image = encoder_loss.item()

            encoder_loss.backward()

            optimizer_encoder.step()

            encoder_avg += encoder_loss_per_image

        # -- batch-loop

        if epoch_idx % cfg.print_every == 0:
            time_end = time.time()
            time_used = time_end - time_start
            logger.debug(
                '[%3d/%3d] avg_loss: %.8f encoder_loss:%.8f used time: %s rest_time: %s' %
                (epoch_idx, cfg.num_epochs, epoch_avg / (cfg.print_every * len(dataloader)),
                 encoder_avg / (cfg.print_every * len(dataloader)),
                 time_change(time_used), time_change(
                    time_used / (epoch_idx - cfg.start_epoch + 1) * (cfg.num_epochs - epoch_idx))))
            writer.add_scalar("train/epoch_avg_loss", epoch_avg / (cfg.print_every * len(dataloader)),
                              epoch_idx // cfg.print_every)
            ts += 1
            epoch_avg = 0.0
            encoder_avg = 0.0

        if epoch_idx % cfg.save_img_every == 0:
            x = Variable(patch[0, :, :, :].unsqueeze(0)).cuda()
            out = model(x).cpu().data
            out = np.reshape(out, (3, 128, 128))
            y = torch.cat((img[0], out), dim=2)
            y = (y + 1) / 2
            save_imgs(imgs=y,
                      name="../experiments/%s/out/out_%05d.png" % (cfg.exp_name, epoch_idx))

        # save model
        if epoch_idx % cfg.save_model_every == 0:
            model.save_model(cfg.exp_name, epoch_idx)

    # -- train-loop

    # save final model
    model.save_final_model(cfg.exp_name)

    # final setup
    epilogue(cfg, writer)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)

    train(config)
