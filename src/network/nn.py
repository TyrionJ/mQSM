from typing import Tuple
import torch.nn.functional as F
from torch import nn
import torch
from einops.layers.torch import Rearrange


def _MinMaxScaler(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    return x


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(1, 1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class ABLayer(nn.Module):
    """
    Attention backward layer
    """
    def __init__(
            self,
            in_channel: int,
            out_channels: Tuple[int, ...],
            kernel_size: Tuple[int, ...],
            stride: Tuple[int, ...]
    ):
        super().__init__()
        modules = []
        for out_ch in out_channels:
            modules.append(ConvDropoutNormReLU(in_channel, out_ch, kernel_size, stride))
        self.skip_convs = nn.ModuleList(modules)
        self.num_layer = len(out_channels)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, skips, attn):
        s_x, x_a = x.shape[2:], attn.shape[2:]
        for i in range(self.num_layer):
            s_sk = skips[i].shape[2:]
            sf1 = [k / j for j, k in zip(s_x, s_sk)]
            sf2 = [k / j for j, k in zip(x_a, s_sk)]
            _x = F.interpolate(x, scale_factor=sf1, mode='trilinear')
            _a = F.interpolate(attn, scale_factor=sf2, mode='trilinear')
            skips[i] = self.nonlin(skips[i] + self.skip_convs[i](_x * _a))
        return skips


class SABlock(nn.Module):
    """
    Self attention block
    """
    def __init__(self, spatial_dims: int, hidden_size: int, num_heads: int = 1, dropout_rate: float = 0.0,
                 qkv_bias: bool = False, save_attn: bool = False) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dims should be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = torch.tensor(self.head_dim ** -0.5)
        self.attn_map = torch.tensor(0.)
        self.save_attn = save_attn

    def forward(self, x):
        if self.spatial_dims == 3:
            _, _, X, Y, _ = x.shape
            x = Rearrange("b c x y z -> b (x y z) c")(x)
        else:
            _, _, X, Y = x.shape
            x = Rearrange("b c x y -> b (x y) c")(x)
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        del x, output
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        del k
        if self.save_attn:
            self.attn_map = Rearrange("b h (x y z) -> b h x y z", x=X, y=Y)(att_mat.detach().mean(dim=2))
            self.attn_map = _MinMaxScaler(self.attn_map)

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        del att_mat
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        if self.spatial_dims == 3:
            x = Rearrange("b (x y z) c -> b c x y z", x=X, y=Y)(x)
        else:
            x = Rearrange("b (x y) c -> b c x y", x=X)(x)

        return x, self.out_rearrange(q), self.out_rearrange(v)


class FIFALayer(nn.Module):
    """
    Feature Instance Fusion Attention
    """
    def __init__(self, feature_num1: int, feature_num2: int, num_heads: int = 1, dropout_rate: float = 0.0,
                 kv_bias: bool = False, save_attn: bool = False):
        super().__init__()

        self.feature_num1 = feature_num1
        self.feature_num2 = feature_num2
        self.sa1 = SABlock(3, feature_num1, dropout_rate=dropout_rate,
                           qkv_bias=kv_bias, save_attn=save_attn, num_heads=num_heads)
        self.sa2 = SABlock(3, feature_num2, dropout_rate=dropout_rate,
                           qkv_bias=kv_bias, save_attn=save_attn, num_heads=num_heads)
        self.out1 = ConvDropoutNormReLU(feature_num1, feature_num1, (3, 3, 3), (1, 1, 1))
        self.out2 = ConvDropoutNormReLU(feature_num2, feature_num2, (3, 3, 3), (1, 1, 1))
        self.attn_map = torch.tensor(0.)
        self.save_attn = save_attn

    def forward(self, x):
        f1, f2 = x[:, :self.feature_num1], x[:, -self.feature_num2:]

        f1, q1, v1 = self.sa1(f1)
        f2, q2, v2 = self.sa2(f2)

        scale = torch.prod(torch.tensor(x.shape[2:])) ** -0.5
        fusion_attn = torch.tanh(torch.einsum("bnC,bnc->bCc", q1, q2) * scale)
        if self.save_attn:
            self.attn_map = _MinMaxScaler(fusion_attn.detach())

        ff1 = torch.einsum("bCc,bnc->bCn", fusion_attn, v2)
        ff2 = torch.einsum("bCc,bnC->bcn", fusion_attn, v1)

        shape_rearrange = Rearrange("b c (x y z) -> b c x y z", x=x.shape[2], y=x.shape[3])
        ff1 = shape_rearrange(ff1)
        ff2 = shape_rearrange(ff2)

        return torch.concatenate([f1, ff1, f2, ff2], dim=1)
