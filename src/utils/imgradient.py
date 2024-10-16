import torch
from torch.nn.functional import conv3d

wx, wy, wz = None, None, None


def imgradient3(data):
    global wx, wy, wz
    if wx is None:
        kx = torch.FloatTensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]).to(device=data.device)
        ky = torch.FloatTensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                 [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).to(device=data.device)
        kz = torch.FloatTensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                 [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]).to(device=data.device)

        wx = kx.unsqueeze(dim=0).unsqueeze(dim=0)
        wy = ky.unsqueeze(dim=0).unsqueeze(dim=0)
        wz = kz.unsqueeze(dim=0).unsqueeze(dim=0)

    dx = conv3d(data, wx, padding=1)
    dy = conv3d(data, wy, padding=1)
    dz = conv3d(data, wz, padding=1)

    return dx, dy, dz
