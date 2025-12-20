import os
import hydra
import torch
import copy
import random
import tqdm
import numpy as np
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from termcolor import cprint
from torch.nn import functional as F
from flow_matching_policy.workspace.base_workspace import BaseWorkspace
from flow_matching_policy.model.common.lr_scheduler import get_scheduler
from flow_matching_policy.common.checkpoint_util import TopKCheckpointManager
from flow_matching_policy.common.json_logger import DualJsonLogger
from flow_matching_policy.common.pytorch_util import dict_apply, optimizer_to
OmegaConf.register_new_resolver("eval", eval, replace=True)
class CFMWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = hydra.utils.instantiate(cfg.policy)
        if cfg.training.use_ema:
            ema_model = copy.deepcopy(model)
        self.model = model
        self.ema_model = ema_model if cfg.training.use_ema else None
        self.device = torch.device(cfg.training.device)
        if torch.cuda.is_available():
            torch.cuda.init()
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self.train_construction()
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                self.load_checkpoint(path=lastest_ckpt_path)
        train_log_path = os.path.join(self.output_dir, 'train_logs.json.txt')
        test_log_path = os.path.join(self.output_dir, 'test_logs.json.txt')
        with DualJsonLogger(train_log_path, test_log_path) as (self.train_json_logger, self.test_json_logger):
            for local_epoch_idx in tqdm.tqdm(range(cfg.training.num_epochs), desc=f"Training"):
                self.train(self.train_dataloader)
                self.policy = self.model
                if cfg.training.use_ema:
                    self.policy = self.ema_model
                self.policy.eval()
                train_skpt_point = self.test_in_dataset(self.eval_train_dataloader)
                val_skpt_point = self.test_in_dataset(self.eval_validation_dataloader)
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    topk_ckpt_path = self.topk_manager.get_ckpt_path(train_skpt_point)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    cprint("检查点已保存", "green")
                self.policy.train()
                self.global_step += 1
                self.epoch += 1

    def train_construction(self):
        cfg = copy.deepcopy(self.cfg)
        dataset = hydra.utils.instantiate(cfg.process_dataset)
        train_dataset = dataset.get_train_dataset()
        validation_dataset = dataset.get_validation_dataset()
        train_dataloader = DataLoader(train_dataset, **cfg.train_dataloader)
        eval_train_dataloader = DataLoader(train_dataset, **cfg.eval_dataloader)
        eval_validation_dataloader = DataLoader(validation_dataset, **cfg.eval_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=-1
        )
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        self.train_dataloader = train_dataloader
        self.eval_train_dataloader = eval_train_dataloader
        self.eval_validation_dataloader = eval_validation_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema = ema if cfg.training.use_ema else None
        self.topk_manager = topk_manager
        self.global_step = 0
        self.epoch = 0
        optimizer_to(self.optimizer, self.device)
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
    def train(self, dataloader):
        cfg = copy.deepcopy(self.cfg)
        train_losses = list()
        for batch_idx, batch in enumerate(dataloader):
            batch = dict_apply(
                batch,
                lambda x: x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
            )
            raw_loss, loss_dict = self.model.compute_loss(batch)
            loss = raw_loss / cfg.training.gradient_accumulate_every
            loss.backward()
            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            if cfg.training.use_ema:
                self.ema.step(self.model)
            train_losses.append(loss_dict["train_loss"])
            is_last_batch = (batch_idx == (len(dataloader) - 1))
            if not is_last_batch:
                self.global_step += 1
        epoch_train_loss = np.mean(train_losses)
        step_log = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'epoch_train_loss': epoch_train_loss,
        }
        self.train_json_logger.log(step_log)
        del step_log
    def test_in_dataset(self, dataloader):
        cfg = copy.deepcopy(self.cfg)
        if (self.epoch % cfg.training.val_every) == 0:
            with torch.no_grad():
                train_losses = list()
                position_losses = list()
                time_interval_losses = list()
                train_t_total = 0.0
                batch_count = 0
                sample_batch = next(iter(dataloader))
                sample_batch = dict_apply(
                    sample_batch,
                    lambda x: x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                )
                _ = self.policy.predict_action(sample_batch)
                torch.cuda.synchronize()
                for batch_idx, batch in enumerate(dataloader):
                    t1 = time.time()
                    batch = dict_apply(
                        batch,
                        lambda x: x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                    )
                    pred_action = self.policy.predict_action(batch)
                    t2 = time.time()
                    delta_t = t2 - t1
                    train_t_total += delta_t
                    target_action = batch['optimal_traj']
                    loss = F.mse_loss(pred_action, target_action, reduction='none')
                    train_losses.append(loss.mean().item())
                    position_losses.append(loss[:, :, 0:3].mean().item())
                    time_interval_losses.append(loss[:, :, 3:].mean().item())
                    batch_count += 1
                train_loss = np.sum(train_losses) / batch_count
                position_loss = np.sum(position_losses) / batch_count
                time_interval_loss = np.sum(time_interval_losses) / batch_count
                train_t_avg = train_t_total / batch_count
                if dataloader == self.eval_train_dataloader:
                    cprint(
                        f"训练集损失: {train_loss:.7f}; 坐标损失: {position_loss:.7f}; 时间间隔损失:{time_interval_loss:.7f};计算时间: {train_t_avg:.7f}",
                        "cyan"
                    )
                if dataloader == self.eval_validation_dataloader:
                    cprint(
                        f"测试集损失: {train_loss:.7f}; 坐标损失: {position_loss:.7f}; 时间间隔损失:{time_interval_loss:.7f};计算时间: {train_t_avg:.7f}",
                        "cyan"
                    )
                step_log = {
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'Training_Loss': train_loss,
                    'Position_Loss': position_loss,
                    'Time_Interval_Loss': time_interval_loss,
                    'Average_Computation_Time': train_t_avg,
                }
                self.test_json_logger.log(step_log)
                del step_log
                skpt_point = {"test_mean_score": -train_loss, 'epoch': self.epoch}
                return skpt_point












































        
       
        

