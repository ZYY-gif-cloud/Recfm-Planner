import numpy as np
import numba
from typing import Optional
from flow_matching_policy.dataset.replay_buffer import ReplayBuffer
class SequenceSampler:
    def __init__(self, replay_buffer: ReplayBuffer):
        super().__init__()
        self.replay_buffer = replay_buffer
    def __len__(self):
        try:
            length_value = len(next(iter(self.replay_buffer.values())))
            self.length_value = length_value
        except (StopIteration, TypeError, AttributeError):
            raise ValueError("回放缓冲区必须具有可迭代值")
        return self.length_value
    def sample_sequence(self, idx):
        result = {key: self.replay_buffer[key][idx] for key in self.replay_buffer.keys()}
        return result
