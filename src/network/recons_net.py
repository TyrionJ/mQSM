import torch.nn as nn


class ReconsNet(nn.Module):
    def __init__(self):
        super().__init__()

        feats = 64
        self.in_conv = nn.Sequential(nn.Conv3d(1, feats//2, 3, 1, 1),
                                     nn.Conv3d(feats//2, feats, 3, 1, 1))
        self.blocks = nn.Sequential(ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats))
        self.out1 = nn.Conv3d(feats, feats//2, 3, 1, 1)
        self.out2 = nn.Conv3d(feats//2, 1, 3, 1, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.blocks(x)
        skip = self.out1(x)
        susc = self.out2(skip)

        return susc, skip


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)

        self.drop = nn.Dropout3d(p=0.2)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        _x = self.act(self.norm1(self.conv1(x)))
        _x = self.drop(_x)
        _x = self.norm2(self.conv2(_x))

        return self.act(_x + x)
