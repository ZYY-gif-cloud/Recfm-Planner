import torch
import numpy as np
import torch.nn as nn
from termcolor import cprint
def shuffle_point_torch(point_cloud):
    indices = torch.randperm(point_cloud.shape[1], device=point_cloud.device)
    return point_cloud[:, indices]
def pad_point_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros((B, num_pad, C), device=device, dtype=point_cloud.dtype)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud
def uniform_sampling_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)
    indices = torch.randperm(N, device=device)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points
class CloudPointsEncoder(nn.Module):
    def __init__(self, cloudpoints_encoder_cfg=None):
        super().__init__()
        cloudpoints_shape = cloudpoints_encoder_cfg.cloudpoints_shape
        hidden_dim = cloudpoints_encoder_cfg.hidden_dim
        output_dim = cloudpoints_encoder_cfg.output_dim
        num_layers = cloudpoints_encoder_cfg.num_layers
        activation = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        conv_input = nn.Conv1d(3, hidden_dim, kernel_size=1)
        layers = nn.ModuleList()
        global_layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
            global_layers.append(nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1))
        conv_output = nn.Conv1d(hidden_dim * num_layers, output_dim, kernel_size=1)
        cprint("------------------------------------------------------------", "yellow")
        cprint(f"[CloudPointsEncoder] 输入维度: {cloudpoints_shape}", "yellow")
        cprint(f"[CloudPointsEncoder] 输出维度: {[output_dim]}", "yellow")
        cprint("------------------------------------------------------------", "yellow")
        self.cloudpoints_shape = cloudpoints_shape
        self.num_points = cloudpoints_shape[0]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.conv_input = conv_input
        self.layers = layers
        self.global_layers = global_layers
        self.conv_output = conv_output
    def output_shape(self):
        return self.output_dim
    def forward(self, cloudpoints):
        points = uniform_sampling_torch(cloudpoints, self.num_points)
        points = points.transpose(1, 2)
        feat_list = []
        y = self.activation(self.conv_input(points))
        for i in range(self.num_layers):
            y = self.activation(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y_global = y_global.expand_as(y)
            y = torch.cat([y, y_global], dim=1)
            y = self.activation(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_output(x)
        x_global = x.max(-1).values
        return x_global




