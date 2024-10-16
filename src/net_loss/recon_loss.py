import torch
import torch.nn as nn
from utils.imgradient import imgradient3
from utils.normalization import mean_std


class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, recon):
        y_hat, rec_data = recon
        x, fld, msk = [rec_data[:, i:i+1] for i in range(rec_data.shape[1])]

        l1 = 10 * x.abs() * (y_hat - x)
        Z = torch.zeros_like(l1)
        l_cycle = self.l1_loss(l1[msk == 1], Z[msk == 1])

        f_grdx, f_grdy, f_grdz = imgradient3(fld * msk)
        y_grdx, y_grdy, y_grdz = imgradient3(y_hat * msk)

        f_grdx, f_grdy, f_grdz = f_grdx[msk == 1], f_grdy[msk == 1], f_grdz[msk == 1]
        y_grdx, y_grdy, y_grdz = y_grdx[msk == 1], y_grdy[msk == 1], y_grdz[msk == 1]

        f_grdx = abs(mean_std(f_grdx))
        f_grdy = abs(mean_std(f_grdy))
        f_grdz = abs(mean_std(f_grdz))

        y_grdx = abs(mean_std(y_grdx))
        y_grdy = abs(mean_std(y_grdy))
        y_grdz = abs(mean_std(y_grdz))

        l_grad = (self.l1_loss(y_grdx, f_grdx)
                  + self.l1_loss(y_grdy, f_grdy)
                  + self.l1_loss(y_grdz, f_grdz)) / 3

        return l_cycle + 0.01 * l_grad
