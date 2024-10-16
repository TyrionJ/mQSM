import torch
import numpy as np
from torch import nn
from typing import Callable

from utils.softmax import softmax_helper_dim1


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, Input, Target):
        if len(Target.shape) == len(Input.shape):
            assert Target.shape[1] == 1
            Target = Target[:, 0]
        return super().forward(Input, Target.long())


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        ce_loss = self.ce(net_output, target[:, 0].long())

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        for i in args:
            assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)

        if self.weight_factors is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(*[j[0] for j in args])
        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                continue
            l += weights[i] * self.loss(*inputs)
        return l


class SegLoss(DeepSupervisionWrapper):
    def __init__(self, ds_scales):
        loss = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {},
                              weight_ce=1, weight_dice=1,
                              ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

        weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
        weights[-1] = 0

        weights = weights / weights.sum()
        super().__init__(loss, weights)

    def forward(self, segm):
        return super().forward(segm[0], segm[1])
