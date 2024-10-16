import torch
from torch import nn

from network.nn import ConvDropoutNormReLU, FIFALayer


class FIFANet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layer = nn.Sequential(
            ConvDropoutNormReLU(in_channels, 32, (3, 3, 3), (1, 1, 1)),
            ConvDropoutNormReLU(32, 32, (3, 3, 3), (1, 1, 1))
        )

        self.encoders = nn.ModuleList([
            nn.Sequential(
                ConvDropoutNormReLU(64, 64, (3, 3, 3), (2, 2, 2), groups=2)),
            nn.Sequential(
                ConvDropoutNormReLU(64, 128, (3, 3, 3), (2, 2, 2), groups=2),
                ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1), groups=2)),
            nn.Sequential(
                FIFALayer(64, 64),
                ConvDropoutNormReLU(256, 256, (3, 3, 3), (2, 2, 2), groups=2),
                ConvDropoutNormReLU(256, 256, (3, 3, 3), (1, 1, 1), groups=2)),
            nn.Sequential(
                FIFALayer(128, 128),
                ConvDropoutNormReLU(512, 512, (3, 3, 3), (2, 2, 2)),
                ConvDropoutNormReLU(512, 512, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(
                ConvDropoutNormReLU(512, 512, (3, 3, 3), (2, 2, 1)),
                ConvDropoutNormReLU(512, 512, (3, 3, 3), (1, 1, 1)))
        ])

        self.stages = nn.ModuleList([
            nn.Sequential(ConvDropoutNormReLU(1024, 512, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(512, 512, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(512, 256, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(256, 256, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(256, 128, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(128, 64, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(64, 64, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(64, 32, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(32, 32, (3, 3, 3), (1, 1, 1))),
        ])

        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv3d(512, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(256, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(128, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        ])
        self.decoder = FakeDecoder()
        print('FIFANet initialized')

    def set_deep_supervision(self, mode):
        self.decoder.deep_supervision = mode

    def forward(self, x):
        x, recon_skip = x
        x = self.in_layer(x)
        skips = [x]
        x = torch.concatenate([recon_skip, x], dim=1)

        t = x
        for encoder in self.encoders:
            t1 = encoder(t)
            if len(torch.where(torch.isnan(t1[0, 0]))[0]) > 0:
                t = encoder(t)
            else:
                t = t1
            skips.append(t)
        del x, t

        seg_outputs = []
        lup_inp = skips[-1]
        for i in range(len(self.stages)):
            x = self.trans_convs[i](lup_inp)
            x = torch.cat((x, skips[-(i + 2)]), 1)
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))
            lup_inp = x
        seg_outputs = seg_outputs[::-1]

        return seg_outputs if self.decoder.deep_supervision else seg_outputs[0]


class FakeDecoder:
    def __init__(self):
        self.deep_supervision = False
