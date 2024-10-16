import torch.nn as nn

from .recon_loss import ReconLoss
from .seg_loss import SegLoss


class NetLoss(nn.Module):
    def __init__(self, ds_scales):
        super().__init__()

        self.recon_loss = ReconLoss()
        self.seg_loss = SegLoss(ds_scales)

    def forward(self, recon, segm):
        r_loss = self.recon_loss(recon)
        s_loss = self.seg_loss(segm)

        return r_loss + s_loss, r_loss, s_loss
