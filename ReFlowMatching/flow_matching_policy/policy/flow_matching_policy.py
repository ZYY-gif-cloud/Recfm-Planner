import torch
import time
from typing import List, Optional, Tuple, Union
from torch.nn import functional as F
from einops import reduce
from termcolor import cprint
import torch.nn as nn
from flow_matching_policy.model.networks.start_end_encoder import StartEndEncoder
from flow_matching_policy.model.networks.init_path_encoder import InitPathEncoder
from flow_matching_policy.model.networks.cloudpoints_encoder import CloudPointsEncoder
from flow_matching_policy.model.networks.timesteps_encoder import TimeStepsEncoder
from flow_matching_policy.model.networks.conditional_unet1d import ConditionalUnet1D
from flow_matching_policy.model.common.normalizer import LinearNormalizer
from flow_matching_policy.model.schedulers.flow_matcher import FlowMatcher
from flow_matching_policy.common.model_util import print_params

class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        cloudpoints_encoder_cfg,
        init_path_encoder_cfg,
        start_end_encoder_cfg,
        timesteps_encoder_cfg,
        conditional_unet1d_cfg,
        flow_matcher_cfg,
        device,
        **kwargs
    ):
        super().__init__()
        start_end_encoder = StartEndEncoder(start_end_encoder_cfg=start_end_encoder_cfg)
        start_end_encoder_dim = start_end_encoder.output_shape()
        cloudpoints_encoder = CloudPointsEncoder(cloudpoints_encoder_cfg=cloudpoints_encoder_cfg)
        cloudpoints_encoder_dim = cloudpoints_encoder.output_shape()
        timesteps_encoder = TimeStepsEncoder(timesteps_encoder_cfg=timesteps_encoder_cfg)
        timesteps_encoder_dim = timesteps_encoder.output_shape()
        global_cond_dim = start_end_encoder_dim + cloudpoints_encoder_dim + timesteps_encoder_dim
        conditionalunet1d = ConditionalUnet1D(conditional_unet1d_cfg=conditional_unet1d_cfg, global_cond_dim=global_cond_dim)
        conditionalunet1d_output_dim = conditionalunet1d.output_shape()
        flow_matcher = FlowMatcher(flow_matcher_cfg=flow_matcher_cfg, device=device)
        normalizer = LinearNormalizer()
        self.device = device
        self.start_end_encoder = start_end_encoder
        self.timesteps_encoder = timesteps_encoder
        self.cloudpoints_encoder = cloudpoints_encoder
        self.conditionalunet1d = conditionalunet1d
        self.flow_matcher = flow_matcher
        self.normalizer = normalizer
        self.conditionalunet1d_output_dim = torch.tensor(conditionalunet1d_output_dim, device=self.device)
        print_params(self)
    def conditional_sample(self, global_feature=None, in_traj=None, **kwargs):
        noise = torch.randn(in_traj.shape, device=self.device)
        trajectory = self.flow_matcher.get_acc_point_traj(in_traj, noise)
        for timestep in self.flow_matcher.infer_timesteps:
            timesteps_encoder_output = self.timesteps_encoder(timestep.expand(trajectory.shape[0]))
            all_feature = torch.cat([global_feature, timesteps_encoder_output], axis=-1)
            model_output = self.conditionalunet1d(sample=trajectory, global_cond=all_feature)
            trajectory = self.flow_matcher.step(trajectory, model_output)
        return trajectory
    def predict_action(self, batch):
        deep_cloud = batch["deep_cloud"]
        start_end = self.normalizer.normalize({"start_end": batch["start_end"]})["start_end"]
        degraded_trajectory = self.normalizer.normalize({"trajectory": batch["init_traj"]})["trajectory"]
        cloudpoints_encoder_output = self.cloudpoints_encoder(deep_cloud)
        if deep_cloud.shape[0] != degraded_trajectory.shape[0]:
            cloudpoints_encoder_output = cloudpoints_encoder_output.repeat(degraded_trajectory.shape[0], 1)
        start_end_encoder_output = self.start_end_encoder(start_end)
        global_feature = torch.cat([cloudpoints_encoder_output, start_end_encoder_output], axis=-1)
        trajectory = self.conditional_sample(global_feature=global_feature, in_traj=degraded_trajectory)
        trajectory = self.normalizer.unnormalize({'trajectory': trajectory})["trajectory"]
        return trajectory
    def compute_loss(self, batch):
        deep_cloud = batch["deep_cloud"]
        start_end = self.normalizer.normalize({"start_end": batch["start_end"]})["start_end"]
        trajectory = self.normalizer.normalize({"trajectory": batch["optimal_traj"]})["trajectory"]
        degraded_trajectory = self.normalizer.normalize({"trajectory": batch["init_traj"]})["trajectory"]
        cloudpoints_encoder_output = self.cloudpoints_encoder(deep_cloud)
        start_end_encoder_output = self.start_end_encoder(start_end)
        noise = torch.randn(trajectory.shape, device=self.device)
        residual = (degraded_trajectory - trajectory)
        train_timesteps = self.flow_matcher.set_train_timesteps(trajectory.shape[0])
        noisy_trajectory = self.flow_matcher.sample_conditional_pt(trajectory, noise, residual, train_timesteps)
        timesteps_encoder_output = self.timesteps_encoder(train_timesteps)
        global_feature = torch.cat([cloudpoints_encoder_output, start_end_encoder_output, timesteps_encoder_output],axis=-1)
        model_output = self.conditionalunet1d(sample=noisy_trajectory, global_cond=global_feature)
        vector_field = self.flow_matcher.compute_conditional_vector_field(trajectory, noise, residual)
        vector_field_loss = self.flow_matcher.compute_loss(model_output, vector_field)
        loss_dict = {'train_loss': vector_field_loss.item()}
        return vector_field_loss, loss_dict
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

 