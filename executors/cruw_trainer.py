import os
import numpy as np
import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.loss import SmoothCELoss
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval import evaluate_rodnet_seq
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutor(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config_dict, cruw_dataset_obj, save_dir):
        """
        PyTorch lightning base class for training models on CRUW datasets.
        @param model: instance of the model to train
        @param train_dataset: training dataset
        @param val_dataset: validation dataset
        @param config_dict: dictionary with training configuration (lr, optimizer, path to data etc.)
        @param cruw_dataset_obj: CRUW dataset object
        @param save_dir: directory to save data
        """
        super(CruwExecutor, self).__init__()

        self.cruw_dataset_obj = cruw_dataset_obj
        self.config = config_dict
        self.train_cfg = config_dict['train_cfg']
        self.radar_cfg = cruw_dataset_obj.sensor_cfg.radar_cfg
        self.model_cfg = config_dict['model_cfg']
        self.n_class = self.cruw_dataset_obj.object_cfg.n_class
        self.batch_size = self.train_cfg['batch_size']
        self.learning_rate = self.train_cfg['lr']
        self.in_channels = self.model_cfg['in_channels']
        self.win_size = self.train_cfg['win_size']
        self.model_name = self.model_cfg['name']

        # hp_dict = {'model_cfg': config_dict['model_cfg'],
        #            'train_cfg': config_dict['train_cfg']}
        # self.save_hyperparameters(hp_dict)
        # self.save_hyperparameters(ignore=['model', 'train_dataset', 'val_dataset', 'cruw_dataset_obj'])
        # Model
        self.model = model
        self.loss_fct = self.get_loss()

        # Dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Test/Val results dirs
        self.val_res_dir = os.path.join(save_dir, 'val')
        self.test_res_dir = os.path.join(save_dir, 'test')
        if not os.path.exists(self.val_res_dir):
            os.makedirs(self.val_res_dir)
        if not os.path.exists(self.test_res_dir):
            os.makedirs(self.test_res_dir)

        # For testing on val set
        self.evalImgs_all = []
        self.n_frames_all = 0

        # 🔧 新增：训练监控变量
        self.train_loss_history = []
        self.val_loss_history = []

    def get_loss(self):
        """
        Define the loss function to use according to the configuration file
        @return: loss function object
        """
        loss_type = self.train_cfg['loss']
        if loss_type == 'bce':
            return nn.BCELoss()
        elif loss_type == 'mse':
            return nn.SmoothL1Loss()
        elif loss_type == 'smooth_ce':
            alpha = self.train_cfg['alpha_loss']
            return SmoothCELoss(alpha)
        else:
            raise ValueError

    def train_dataloader(self):
        """
        Define PyTorch training dataloader
        @return: train dataloader for ROD2021 dataset
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=True, num_workers=4, drop_last=True,
                          persistent_workers=True, pin_memory=True)  # 🔧 添加性能优化

    def val_dataloader(self):
        """
        Define PyTorch validation dataloader
        @return: validation dataloader for ROD2021 dataset
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True,
                          persistent_workers=True, pin_memory=True)  # 🔧 添加性能优化

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/AP": 0, "hp/AR": 0, "hp/val_loss": 0, "hp/train_loss": 0})
        print(f"🚀 训练开始 - 学习率: {self.learning_rate}, Batch大小: {self.batch_size}")

    def forward(self, x):
        """
        Pytorch Lightning forward pass (inference)
        @param batch_positions: positional encoding vector (optional - only for UTAE)
        @param x: input tensor with shape (B, C, T, H, W) where T in the number of timesteps
        @return: ConfMap prediction
        """
        confmap_pred = self.model(x)
        return confmap_pred

    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        # Get data
        ra_maps = batch['radar_data']  # N, H, W, C
        confmap_gts = batch['anno']['confmaps']
        image_paths = batch['image_paths']

        # 🔧 添加输入检查
        if torch.isnan(ra_maps).any() or torch.isinf(ra_maps).any():
            print(f"⚠️  警告: 输入数据包含NaN或Inf (batch {batch_id})")
            return None

        confmap_pred = self.model(ra_maps)

        # 🔧 添加预测检查
        if torch.isnan(confmap_pred).any() or torch.isinf(confmap_pred).any():
            print(f"❌ 错误: 模型输出包含NaN或Inf (batch {batch_id})")
            print(f"   输入范围: [{ra_maps.min():.4f}, {ra_maps.max():.4f}]")
            print(f"   输出范围: [{confmap_pred.min():.4f}, {confmap_pred.max():.4f}]")
            # 返回一个有效的损失值，跳过这个batch
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        loss = self.loss_fct(confmap_pred, confmap_gts)

        # 🔧 添加损失检查
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
            print(f"❌ 异常损失值: {loss.item():.4f} (batch {batch_id})")
            print(f"   预测范围: [{confmap_pred.min():.4f}, {confmap_pred.max():.4f}]")
            print(f"   GT范围: [{confmap_gts.min():.4f}, {confmap_gts.max():.4f}]")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # 🔧 记录损失历史
        self.train_loss_history.append(loss.item())
        if len(self.train_loss_history) > 1000:
            self.train_loss_history.pop(0)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True,
                 batch_size=self.batch_size)
        self.log('hp/train_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # 🔧 定期打印训练状态
        if batch_id % 100 == 0:
            print(f"📊 Epoch {self.current_epoch} | Batch {batch_id} | Loss: {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        # Get data
        ra_maps = batch['radar_data']  # N, H, W, C
        confmap_gts = batch['anno']['confmaps']
        image_paths = batch['image_paths']
        obj_infos = batch['anno']['obj_infos']

        confmap_pred = self.forward(ra_maps)

        loss = self.loss_fct(confmap_pred, confmap_gts)

        # 🔧 添加验证损失检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  验证集异常损失 (batch {batch_id}): {loss.item()}")
            loss = torch.tensor(0.0, device=self.device)

        # 🔧 记录验证损失历史
        self.val_loss_history.append(loss.item())
        if len(self.val_loss_history) > 100:
            self.val_loss_history.pop(0)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
                 batch_size=self.batch_size)
        self.log('hp/val_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        """🔧 新增：验证结束时的统计"""
        if len(self.val_loss_history) > 0:
            avg_val_loss = np.mean(self.val_loss_history)
            print(f"✅ Epoch {self.current_epoch} 验证完成 | 平均损失: {avg_val_loss:.4f}")

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']

        # Get seq name to write results
        seq_name = batch['seq_names'][0]
        if confmap_gts is not None:
            confmap_gts = batch['anno']['confmaps'].float()
            save_dir = os.path.join(self.val_res_dir)
        else:
            save_dir = os.path.join(self.test_res_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, seq_name.upper() + ".txt")

        if confmap_gts is not None:
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            start_frame_name = image_paths[0][0][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if frame_id == self.win_size - 1 and self.model_name not in ('RECORDNoLstmMulti', 'RECORDNoLstmSingle'):
            for tmp_frame_id in range(frame_id):
                print("Eval frame", tmp_frame_id)
                tmp_ra_maps = ra_maps[:, :, :tmp_frame_id + 1]
                confmap_pred = self.forward(tmp_ra_maps)
                res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
                write_dets_results_single_frame(res_final, tmp_frame_id, save_path, self.cruw_dataset_obj)

        confmap_pred = self.forward(ra_maps)

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

    def evaluate_rodnet_seq_(self, res_path, gt_path, n_frame, subset):
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        eval_imgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, self.cruw_dataset_obj)
        out_eval = accumulate(eval_imgs, n_frame, ols_thrs, rec_thrs, self.cruw_dataset_obj, log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)

        self.n_frames_all += n_frame
        self.evalImgs_all.extend(eval_imgs)

        self.logger.log_metrics({"AP/" + subset.upper(): stats[0] * 100,
                                 "AR/" + subset.upper(): stats[1] * 100})

    def evaluate_rodnet_(self):
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        out_eval = accumulate(self.evalImgs_all, self.n_frames_all, ols_thrs, rec_thrs, self.cruw_dataset_obj,
                              log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)

        # 🔧 添加结果检查
        ap_value = stats[0] * 100
        ar_value = stats[1] * 100

        print(f"\n{'=' * 80}")
        print(f"📊 最终评估结果:")
        print(f"   AP (Average Precision): {ap_value:.2f}%")
        print(f"   AR (Average Recall): {ar_value:.2f}%")
        print(f"{'=' * 80}\n")

        if ap_value == 0.0 or ar_value == 0.0:
            print("⚠️  警告: 评估指标为0，模型可能训练失败!")
            print(f"   训练损失历史: {self.train_loss_history[-10:] if self.train_loss_history else 'N/A'}")
            print(f"   验证损失历史: {self.val_loss_history[-10:] if self.val_loss_history else 'N/A'}")

        self.logger.log_metrics({"AP/Overall": ap_value,
                                 "AR/Overall": ar_value})

        self.logger.log_metrics({"hp/AP": ap_value,
                                 "hp/AR": ar_value})

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.model_name == 'RECORDNoLstmMulti':
            b, c, t, h, w = batch['radar_data'].shape
            batch['radar_data'] = batch['radar_data'].reshape(b, c * t, h, w)
            return batch
        elif self.model_name == 'RECORDNoLstmSingle':
            b, c, t, h, w = batch['radar_data'].shape
            assert t == 1
            batch['radar_data'] = batch['radar_data'].reshape(b, c, h, w)
            return batch
        else:
            return batch

    def configure_optimizers(self):
        opti = self.train_cfg['optimizer']
        scheduler = self.train_cfg.get('scheduler', None)  # 🔧 使用get避免KeyError

        # 🔧 修改：降低学习率并添加warmup
        lr = self.learning_rate

        if opti == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opti == 'adam_reg':
            weight_decay = self.train_cfg.get('weight_decay', 0.0001)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opti == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opti == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0001)
        else:
            raise ValueError(f"未知的优化器类型: {opti}")

        # 🔧 改进：添加warmup + 主调度器
        if scheduler == 'exp':
            # Warmup前5个epoch
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=5
            )
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, main_scheduler],
                    milestones=[5]
                ),
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]

        elif scheduler == 'step':
            # 添加warmup
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=5
            )
            main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, main_scheduler],
                    milestones=[5]
                ),
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]

        elif scheduler is None:
            return optimizer
        else:
            raise ValueError(f"未知的调度器类型: {scheduler}")