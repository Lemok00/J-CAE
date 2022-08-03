import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )
        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )
        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )
        # fully connect 128
        self.fc_128 = nn.Sequential(
            nn.Linear(in_features=32 * 32 * 32, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=128),
            nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)

        batch_size = ec3.shape[0]

        fc128 = self.fc_128(ec3.reshape(shape=(batch_size, 32 * 32 * 32)))

        # Binarisation
        with torch.no_grad():
            rand = torch.rand(fc128.shape).cuda()
            prob = (1 + fc128) / 2
            eps = torch.zeros(fc128.shape).cuda()
            eps[rand <= prob] = (1 - fc128)[rand <= prob]
            eps[rand > prob] = (-fc128 - 1)[rand > prob]

        encoded = 0.5 * (fc128 + eps + 1)  # (-1|1) -> (0|1)
        return encoded

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 32x32x32
        self.d_fc128 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=32 * 32 * 32),
            nn.Tanh()
        )
        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )
        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )
        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )
        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        y = x * 2.0 - 1  # (0|1) -> (-1|1)

        batch_size = y.shape[0]
        d32x32x32 = self.d_fc128(y).reshape(shape=(batch_size, 32, 32, 32))

        uc1 = self.d_up_conv_1(d32x32x32)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        decoded = self.d_up_conv_3(uc2)

        return decoded

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def save_model(self, exp_name, epoch_idx):
        torch.save(self.encoder.state_dict(), f"../experiments/{exp_name}/chkpt/encoder_{epoch_idx}.pth")
        torch.save(self.decoder.state_dict(), f"../experiments/{exp_name}/chkpt/decoder_{epoch_idx}.pth")

    def save_final_model(self, exp_name):
        torch.save(self.encoder.state_dict(), f"../experiments/{exp_name}/chkpt/encoder_final.pth")
        torch.save(self.decoder.state_dict(), f"../experiments/{exp_name}/chkpt/decoder_final.pth")

    def load_model(self, encode, decode):
        self.encoder.load_state_dict(torch.load(encode))
        self.decoder.load_state_dict(torch.load(decode))

    def encode(self,x):
        return self.encoder(x)

    def decode(self,x):
        return self.decoder(x)


