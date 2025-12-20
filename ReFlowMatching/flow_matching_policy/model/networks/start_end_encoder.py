import torch
import math
import torch.nn as nn
import numpy as np
from typing import Union, Dict, List, Type
from termcolor import cprint
import time
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
class StartEndEncoder(nn.Module):
    def __init__(self, start_end_encoder_cfg):
        super().__init__()
        start_end_shape = start_end_encoder_cfg.start_end_shape
        state_mlp_size = start_end_encoder_cfg.state_mlp_size
        state_mlp_activation_fn = start_end_encoder_cfg.state_mlp_activation_fn
        cprint("------------------------------------------------------------", "yellow")
        cprint(f"[StartEndEncoder] 输入维度: {start_end_shape}", "yellow")
        state_shape = np.prod(start_end_shape)
        if not state_mlp_size:
            raise ValueError("MLP维度配置为空")
        net_arch = state_mlp_size[:-1] if len(state_mlp_size) > 1 else []
        output_dim = state_mlp_size[-1]
        activation_function_map = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "Mish": nn.Mish}
        activation_fn_class = activation_function_map.get(state_mlp_activation_fn)
        state_mlp = nn.Sequential(*create_mlp(state_shape, output_dim, net_arch, activation_fn_class))
        cprint(f"[StartEndEncoder] 输出维度: {[output_dim]}", "yellow")
        cprint("------------------------------------------------------------", "yellow")
        self.state_mlp = state_mlp
        self.output_dim = output_dim
    def output_shape(self):
        return self.output_dim
    def forward(self, observations: Union[torch.Tensor, float, int, dict, list]) -> torch.Tensor:
        state = observations.reshape(observations.shape[0], -1)
        state_feat = self.state_mlp(state)
        return state_feat
def get_positional_encoding(observations):
    device = observations.device
    d_model = observations.shape[1]
    div_term = torch.exp(torch.arange(0, d_model, 1, device=device).float() * (-math.log(10000.0) / d_model)).unsqueeze(0).unsqueeze(2).to(device)
    observations = observations * div_term.expand_as(observations)
    return observations
