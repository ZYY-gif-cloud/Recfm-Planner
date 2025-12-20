import pickle
import zarr
import numpy as np
from termcolor import cprint
from typing import Union, Dict

class ReplayBuffer:
    def __init__(self, root: Union[zarr.Group, Dict[str, dict]] = dict(), val_ratio: float = 0.0, seed=42):
        super().__init__()
        assert ('deep_cloud' in root)
        assert ('start_end' in root)
        assert ('init_traj' in root)
        assert ('optimal_traj' in root)
        root_lengths = [len(value) for value in root.values()]
        length_value = len(next(iter(root.values())))
        assert all(length == length_value for length in root_lengths)
        self.root = root
        if val_ratio > 0.0:
            mask = np.zeros(length_value, dtype=bool)
            np.random.seed(seed)
            random_indices = np.random.choice(length_value, int(length_value * val_ratio), replace=False)
            mask[random_indices] = True
            self.val_mask = mask
            self.train_mask = ~self.val_mask
            cprint(f"训练数据量: {self.train_mask.sum()}, 验证数据量: {self.val_mask.sum()}", "green")
    def keys(self):
        return self.root.keys()
    def values(self):
        return self.root.values()
    def items(self):
        return self.root.items()
    def __getitem__(self, key):
        return self.root[key]
    def __contains__(self, key):
        return key in self.root
    @classmethod
    def copy_from_path(cls, zarr_path, val_ratio: float = 0.0, seed=42, **kwargs):
        try:
            with open(zarr_path, 'rb') as file:
                root = pickle.load(file)
            buffer = cls(root=root, val_ratio=val_ratio, seed=seed)
            for key, value in buffer.items():
                try:
                    cprint(
                        f'回放缓冲区: {key}, 形状 {value.shape}, 数据类型 {value.dtype}, 范围 {value.min():.2f}~{value.max():.2f}',
                        'green'
                    )
                except Exception:
                    cprint(
                        f'回放缓冲区: {key}, 形状 {getattr(value, "shape", "未知")}, 数据类型 {getattr(value, "dtype", "未知")}',
                        'green'
                    )

            cprint("-------------------------------------------------------------", 'green')
            return buffer
        except FileNotFoundError:
            cprint(f"文件 {zarr_path} 未找到", "red")
            return None
        except pickle.PickleError:
            cprint(f"无法从文件 {zarr_path} 中加载 pickle 数据", "red")
            return None
        except Exception as e:
            cprint(f"读取文件 {zarr_path} 时发生错误: {e}", "red")
            return None

