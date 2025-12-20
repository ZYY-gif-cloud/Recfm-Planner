import math
import torch
import torch.nn as nn
from typing import Union, Dict, List, Type
from termcolor import cprint
def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU, squash_output: bool = False) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class TimeStepsEncoder(nn.Module):
    def __init__(self, timesteps_encoder_cfg=None):
        super().__init__()
        dsed = timesteps_encoder_cfg.timesteps_embed_dim
        state_mlp_size = timesteps_encoder_cfg.state_mlp_size
        state_mlp_activation_fn = timesteps_encoder_cfg.state_mlp_activation_fn
        cprint("------------------------------------------------------------", "yellow")
        cprint("[TimeStepsEncoder] 输入维度: [1]", "yellow")
        if not state_mlp_size:
            raise ValueError("MLP维度配置为空")
        net_arch = state_mlp_size[:-1] if len(state_mlp_size) > 1 else []
        output_dim = state_mlp_size[-1]
        activation_function_map = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "Mish": nn.Mish}
        activation_fn_class = activation_function_map.get(state_mlp_activation_fn)
        state_mlp = nn.Sequential(*create_mlp(dsed, output_dim, net_arch, activation_fn_class))
        pos_emb = SinusoidalPosEmb(dsed)
        cprint(f"[TimeStepsEncoder] 输出维度: {[output_dim]}", "yellow")
        cprint("------------------------------------------------------------", "yellow")
        self.state_mlp = state_mlp
        self.pos_emb = pos_emb
        self.output_dim = output_dim
    def output_shape(self):
        return self.output_dim
    def forward(self, timestep: Union[torch.Tensor, float, int]) -> torch.Tensor:
        step_feat = self.pos_emb(timestep)
        step_feat = self.state_mlp(step_feat)
        return step_feat

        