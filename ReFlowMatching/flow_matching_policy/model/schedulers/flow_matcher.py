import math
import numpy as np
import torch
from termcolor import cprint
from typing import List, Optional, Tuple, Union
from torch.nn import functional as F
import time
class FlowMatcher:
    def __init__(self, flow_matcher_cfg=None, device=None, **kwargs):
        num_train_timesteps = flow_matcher_cfg.num_train_timesteps
        num_infer_timesteps = flow_matcher_cfg.num_infer_timesteps
        acc_n = flow_matcher_cfg.acc_n
        start_time = flow_matcher_cfg.start_time
        end_time = flow_matcher_cfg.end_time
        self.device = device
        self.num_train_timesteps = torch.tensor(num_train_timesteps).to(device)
        self.num_infer_timesteps = torch.tensor(num_infer_timesteps).to(device)
        self.acc_n = torch.tensor(acc_n).to(device)
        self.start_time = torch.tensor(start_time).to(device)
        self.end_time = torch.tensor(end_time).to(device)
        self.delta_time = ((self.end_time - self.start_time) / num_train_timesteps).to(device) if num_train_timesteps > 0 else torch.tensor(0.0).to(device)
        self.timesteps = (torch.arange(0, self.num_train_timesteps, device=self.device) * self.delta_time + self.start_time)
        self.acc_point = (torch.abs(self.timesteps - self.acc_n / (self.acc_n + 1)).argmin()).to(device)
        self.acc_timestep = self.timesteps[self.acc_point].to(device)
        self.infer_delta_time = ((self.end_time - self.acc_timestep) / num_infer_timesteps).to(device) if num_infer_timesteps > 0 else torch.tensor(0.0).to(device)
        self.infer_timesteps = (torch.arange(0, self.num_infer_timesteps, device=self.device) * self.infer_delta_time + self.acc_timestep)
        cprint(f"------------------------------------------------------------", "red")
        cprint(f"起始时间为：{self.start_time:.6f}；结束时间为：{self.end_time:.6f}；加速系数为：{self.acc_n:.6f}", "red")
        cprint(f"训练时间步数为：{self.num_train_timesteps:.6f}; 训练时间步长为：{self.delta_time:.6f}", "red")
        cprint(f"推理时间步数为：{self.num_infer_timesteps:.6f}; 推理时间步长为：{self.infer_delta_time:.6f}", "red")
        cprint(f"加速截断点为：{self.acc_point:.6f}；加速截断时间步为：{self.acc_timestep:.6f}；误差为：{torch.abs(self.acc_timestep - self.acc_n/(self.acc_n+1)):.6f}", "red")
        cprint(f"------------------------------------------------------------", "red")
    def set_train_timesteps(self, batch_size):
        train_timesteps = torch.randint(self.acc_point, self.num_train_timesteps, (batch_size,), device=self.device) * self.delta_time + self.start_time
        return train_timesteps
    def set_infer_timesteps(self):
        t1 = time.time()
        print(self.acc_point.device)
        infer_timesteps = self.timesteps[self.acc_point:]
        t2 = time.time()
        print(t2 - t1)
        return infer_timesteps
    def get_acc_point_traj(self, degraded_samples: torch.FloatTensor, noise: torch.FloatTensor) -> torch.FloatTensor:
        noisy_samples = (1 - self.acc_timestep) * degraded_samples * self.acc_n + (1 - self.acc_timestep) * noise
        return noisy_samples
    def sample_conditional_pt(self, target_samples: torch.FloatTensor, noise: torch.FloatTensor, residual: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        timestep = timestep.reshape(-1, *([1] * (target_samples.dim() - 1)))
        noisy_samples = timestep * target_samples + (1 - timestep) * noise + self.acc_n * (1 - timestep) * residual
        return noisy_samples
    def compute_conditional_vector_field(self, target_samples: torch.FloatTensor, noise: torch.FloatTensor, residual: torch.FloatTensor) -> torch.FloatTensor:
        vector_field = target_samples - noise - self.acc_n * residual
        return vector_field
    def step(self, sample: torch.FloatTensor, vector_field: torch.IntTensor) -> torch.FloatTensor:
        prev_sample = sample + self.infer_delta_time * vector_field * (self.acc_n + 1)
        return prev_sample
    def compute_loss(self, model_output: torch.FloatTensor, vector_field: torch.IntTensor) -> torch.FloatTensor:
        loss = F.mse_loss(model_output, vector_field / (self.acc_n + 1), reduction='none')
        return loss.mean()











