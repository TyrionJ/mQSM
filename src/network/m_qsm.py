import torch
from torch import nn

from .recons_net import ReconsNet
from .fifa_net import FIFANet


class MQSMNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=31, early_epoch=10):
        super().__init__()

        self.recon_early = early_epoch
        self.recon_end = 150
        self.recons_net = ReconsNet()
        self.seg_net = FIFANet(in_channels-1, num_classes)
        self.recons_net.eval()
        self.seg_net.eval()
        self.cur_epoch = 0

    def set_deep_supervision(self, mode):
        self.seg_net.set_deep_supervision(mode)

    def train(self, epoch=0):
        self.cur_epoch = epoch
        self.recons_net.train()
        self.seg_net.train()
        self.seg_net.set_deep_supervision(True)

    def forward(self, X):
        if self.cur_epoch < self.recon_end:
            susc, skip = self.recons_net(X[:, :1])
        else:
            with torch.no_grad():
                susc, skip = self.recons_net(X[:, :1])
        if self.cur_epoch < self.recon_early:
            with torch.no_grad():
                segm = self.seg_net((X[:, 1:], skip.detach().clone()))
        else:
            segm = self.seg_net((X[:, 1:], skip.detach().clone()))

        return susc, segm
