import torch
from torch.fft import fftshift


def create_kernel(shape, ori_vec=None, vox_sz=None, device=None) -> torch.Tensor:
    if vox_sz is None:
        vox_sz = [1., 1., 1.]
    if ori_vec is None:
        ori_vec = [0., 0., 1.]
    if device is None:
        device = torch.device('cpu')
    N = torch.tensor(shape, dtype=torch.int, device=device)
    ky, kx, kz = torch.meshgrid(torch.arange(-N[0] // 2, N[0] // 2, device=device),
                                torch.arange(-N[1] // 2, N[1] // 2, device=device),
                                torch.arange(-N[2] // 2, N[2] // 2, device=device),
                                indexing="ij")

    spatial = torch.tensor(vox_sz, dtype=torch.float)
    kx = (kx / kx.abs().max()) / spatial[0]
    ky = (ky / ky.abs().max()) / spatial[1]
    kz = (kz / kz.abs().max()) / spatial[2]
    k2 = kx ** 2 + ky ** 2 + kz ** 2 + 2.2204e-16

    tk = 1 / 3 - (kx * ori_vec[0] + ky * ori_vec[1] + kz * ori_vec[2]) ** 2 / k2
    tk = fftshift(tk)

    return tk
