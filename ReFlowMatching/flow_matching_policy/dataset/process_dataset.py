import torch
import numpy as np
import copy
from termcolor import cprint
from typing import Dict
from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.dataset.replay_buffer import ReplayBuffer
from flow_matching_policy.dataset.sampler import SequenceSampler
from flow_matching_policy.model.common.normalizer import LinearNormalizer

class ProcessDataset:
    def __init__(self, zarr_path, seed=42, val_ratio=0.0, **kwargs):
        super().__init__()
        cprint("-------------------------------------------------------------", "green")
        cprint(f'正在加载数据集...', 'green')
        cprint(f'数据集路径：{zarr_path}', 'green')
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, val_ratio=val_ratio, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer)
    def get_validation_dataset(self):
        val_mask = self.replay_buffer.val_mask
        val_buffer = dict()
        for key, value in self.replay_buffer.items():
            val_buffer[key] = value[val_mask]
        val_dataset = copy.deepcopy(self)
        val_dataset.sampler = SequenceSampler(replay_buffer=val_buffer)
        return val_dataset
    def get_train_dataset(self):
        train_mask = self.replay_buffer.train_mask
        train_buffer = dict()
        for key, value in self.replay_buffer.items():
            train_buffer[key] = value[train_mask]
        train_dataset = copy.deepcopy(self)
        train_dataset.sampler = SequenceSampler(replay_buffer=train_buffer)
        return train_dataset
    def get_normalizer(self, mode='limits', **kwargs):
        trajectory = np.concatenate((self.replay_buffer["init_traj"], self.replay_buffer["optimal_traj"]), axis=0)
        data = {
            "deep_cloud": self.replay_buffer["deep_cloud"],
            "start_end": self.replay_buffer["start_end"],
            "trajectory": trajectory
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=mode, **kwargs)
        return normalizer
    def __len__(self) -> int:
        return len(self.sampler)
    def _sample_to_data(self, sample):
        data = {
            "deep_cloud": sample["deep_cloud"].astype(np.float32),
            "start_end": sample["start_end"].astype(np.float32),
            "init_traj": sample["init_traj"].astype(np.float32),
            "optimal_traj": sample["optimal_traj"].astype(np.float32),
        }
        return data
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
