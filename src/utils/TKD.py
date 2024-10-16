import torch
import numpy as np
from torch.fft import fftn, ifftn

from utils.imgradient import imgradient3
from utils.kernel import create_kernel
from utils.normalization import min_max


def cal_kTKD(field, ori=None, thr=None, voxel_size=None):
    if voxel_size is None:
        voxel_size = [1., 1., 1.]
    if ori is None:
        ori = [0., 0., 1.]
    if thr is None:
        thr = 0.2

    kernel = create_kernel(field.shape, ori, voxel_size, field.device)
    ill = kernel.abs() < thr
    kernel[ill] = thr * torch.sign(kernel[ill])
    kernel[kernel == 0] = thr

    k_field = fftn(field)
    k_tkd = k_field / kernel

    return k_tkd


def cal_TKD(field, ori=None, thr=None, voxel_size=None, mask=None):
    kTKD = cal_kTKD(field, ori, thr, voxel_size)

    TKD = ifftn(kTKD)
    TKD = torch.abs(TKD) * torch.sign(torch.real(TKD))

    if mask is not None:
        TKD *= mask

    return TKD


def create_mix(field, ori_vec, voxel_size, mask):
    is_np = isinstance(field, np.ndarray)
    if is_np:
        field = torch.from_numpy(field).float()

    sx, sy, sz = imgradient3(field.unsqueeze(dim=0).unsqueeze(dim=0))
    mag = abs(sx) + abs(sy) + abs(sz)
    mag = min_max(mag[0, 0])

    TKD1 = cal_TKD(field, ori_vec, 0.1, voxel_size, mask)
    TKD25 = cal_TKD(field, ori_vec, 0.25, voxel_size, mask)

    MIX = mag * TKD1 + (1-mag) * TKD25

    kernel = create_kernel(field.shape, ori_vec, voxel_size)
    pos = kernel.abs() < 0.1
    kMIX = fftn(MIX)
    kMIX[pos] *= torch.pow(abs(kernel[pos]), 0.2) * torch.sign(kernel[pos])
    MIX = ifftn(kMIX)
    MIX = torch.abs(MIX) * torch.sign(torch.real(MIX)) * mask

    return MIX.numpy() if is_np else MIX
