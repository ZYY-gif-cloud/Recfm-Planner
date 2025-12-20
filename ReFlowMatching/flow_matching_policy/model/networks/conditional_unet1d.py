import logging
import torch
import torch.nn as nn
import numpy as np
import einops
from einops.layers.torch import Rearrange
from typing import Union, Type
from termcolor import cprint
logger = logging.getLogger(__name__)
class Downsample1d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, stride, padding)
    def forward(self, x):
        return self.conv(x)
class Upsample1d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size, stride, padding)
    def forward(self, x):
        return self.conv(x)
class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, dropout_rate=0.1, activation_fn: Type[nn.Module] = nn.Mish):
        super().__init__()
        assert (kernel_size % 2 != 0)
        assert (out_channels % n_groups == 0)
        self.block = nn.Sequential(nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2), nn.GroupNorm(n_groups, out_channels), activation_fn())
    def forward(self, x):
        return self.block(x)
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, activation_fn: Type[nn.Module] = nn.Mish):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups, activation_fn=activation_fn), Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups, activation_fn=activation_fn)])
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(activation_fn(), nn.Linear(cond_dim, cond_channels), Rearrange('batch t -> batch t 1'))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.out_channels = out_channels
    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
class ConditionalUnet1D(nn.Module):
    def __init__(self, conditional_unet1d_cfg=None, global_cond_dim=None):
        super().__init__()
        down_dims = conditional_unet1d_cfg.down_dims
        kernel_size = conditional_unet1d_cfg.kernel_size
        n_groups = conditional_unet1d_cfg.n_groups
        out_traj_shape = conditional_unet1d_cfg.out_traj_shape
        conditional_unet1d_activation_fn = conditional_unet1d_cfg.conditional_unet1d_activation_fn
        activation_function_map = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "Mish": nn.Mish}
        activation_fn_class = activation_function_map.get(conditional_unet1d_activation_fn)
        all_dims = [out_traj_shape[1]] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        start_dim = down_dims[0]
        out_dim = all_dims[0]
        cond_dim = global_cond_dim
        mid_dim = all_dims[-1]
        cprint("------------------------------------------------------------", "yellow")
        cprint(f"[ConditionalUnet1D] 轨迹输入维度: {out_traj_shape}", "yellow")
        cprint(f"[ConditionalUnet1D] 条件输入维度: {[cond_dim]}", "yellow")
        cprint(f"[ConditionalUnet1D] 输出维度: {[out_traj_shape[0], out_traj_shape[1]]}", "yellow")
        cprint("------------------------------------------------------------", "yellow")
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class)])
        down_modules = nn.ModuleList([])
        downsample1d_dims = [out_traj_shape[0]]
        downsample1d_kernel_size = 3
        downsample1d_stride = 2
        downsample1d_padding = 1
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), Downsample1d(dim_out, kernel_size=downsample1d_kernel_size, stride=downsample1d_stride, padding=downsample1d_padding) if not is_last else nn.Identity()]))
            if not is_last:
                downsample1d_dims.append((downsample1d_dims[-1] + 2 * downsample1d_padding - downsample1d_kernel_size) // downsample1d_stride + 1)
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            upsample1d_stride = downsample1d_stride
            upsample1d_padding = downsample1d_padding
            upsample1d_kernel_size = downsample1d_dims[-ind - 2] + 2 * upsample1d_padding - (downsample1d_dims[-ind - 1] - 1) * upsample1d_stride
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), Upsample1d(dim_in, kernel_size=upsample1d_kernel_size, stride=upsample1d_stride, padding=upsample1d_padding) if not is_last else nn.Identity()]))
        final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups, activation_fn=activation_fn_class), nn.Conv1d(start_dim, out_dim, 1))
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.output_dim = [out_traj_shape[0], out_traj_shape[1]]
    def forward(self, sample: torch.Tensor, global_cond: Union[torch.Tensor, list, np.ndarray], **kwargs):
        sample = einops.rearrange(sample, 'b h t -> b t h')
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_cond)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
    def output_shape(self):
        return self.output_dim

