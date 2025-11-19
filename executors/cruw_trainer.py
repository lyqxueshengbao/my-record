# 完整文件: executors/cruw_trainer.py
import os
import numpy as np
import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.loss import SmoothCELoss, FocalLoss, SmoothFocalLoss
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval import evaluate_rodnet_seq
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutor(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config_dict, cruw_dataset_obj, save_dir):
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
        self.model = model
        self.loss_fct = self.get_loss()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_res_dir = os.path.join(save_dir, 'val')
        self.test_res_dir = os.path.join(save_dir, 'test')
        if not os.path.exists(self.val_res_dir):
            os.makedirs(self.val_res_dir)
        if not os.path.exists(self.test_res_dir):
            os.makedirs(self.test_res_dir)
        self.evalImgs_all = []
        self.n_frames_all = 0

        # TBPTT 状态变量
        self.train_h_state = None
        self.train_c_state = None
        self.last_seq_names = None

    def get_loss(self):
        loss_type = self.train_cfg['loss']
        if loss_type == 'bce':
            return nn.BCELoss()
        elif loss_type == 'focal':
            alpha = self.train_cfg.get('focal_alpha', 0.25)
            gamma = self.train_cfg.get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'smooth_focal':
            alpha = self.train_cfg.get('focal_alpha', 0.25)
            gamma = self.train_cfg.get('focal_gamma', 2.0)
            alpha_weight = self.train_cfg.get('alpha_loss', 0.5)
            return SmoothFocalLoss(alpha=alpha, gamma=gamma, alpha_weight=alpha_weight)
        elif loss_type == 'mse':
            return nn.SmoothL1Loss()
        elif loss_type == 'smooth_ce':
            alpha = self.train_cfg['alpha_loss']
            return SmoothCELoss(alpha)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def train_dataloader(self):
        # Shuffle=False for TBPTT
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/AP": 0, "hp/AR": 0, "hp/val_loss": 0, "hp/train_loss": 0})

    def forward(self, x):
        """
        Inference forward. Handles unpacking and dimension squeezing.
        """
        out = self.model(x)

        # 1. Unpack tuple if necessary
        if isinstance(out, tuple):
            confmap_pred = out[0]
        else:
            confmap_pred = out

        # 2. Squeeze time dimension if T=1 (for Online Inference / Validation compatibility)
        # Record model returns (B, C, T, H, W)
        # RecordOI model returns (B, C, H, W) -> No squeeze needed
        # We only squeeze if it is 5D and T=1.
        if confmap_pred.dim() == 5 and confmap_pred.shape[2] == 1:
            confmap_pred = confmap_pred.squeeze(2)

        return confmap_pred

    def on_train_epoch_start(self):
        self.train_h_state = None
        self.train_c_state = None
        self.last_seq_names = None

    def training_step(self, batch, batch_id):
        ra_maps = batch['radar_data']  # (B, C, T, H, W)
        confmap_gts = batch['anno']['confmaps']  # (B, C, T, H, W) if all_confmaps=True

        # === TBPTT: Sequence Switch Check ===
        current_seq_names = batch['seq_names']
        if self.last_seq_names is not None:
            # If sequence changed, reset state
            if current_seq_names[0] != self.last_seq_names[0]:
                self.train_h_state = None
                self.train_c_state = None
        self.last_seq_names = current_seq_names

        # === TBPTT: Detach States ===
        if self.train_h_state is not None:
            h_state = [h.detach() for h in self.train_h_state]
            c_state = [c.detach() for c in self.train_c_state]
        else:
            h_state = None
            c_state = None

        # Forward
        # Returns (preds, next_h, next_c)
        # preds shape: (B, C, T, H, W)
        confmap_pred, next_h, next_c = self.model(ra_maps, h_state, c_state)

        self.train_h_state = next_h
        self.train_c_state = next_c

        # Compute Loss (Many-to-Many)
        # Ensure shapes match. Both should be (B, C, T, H, W)
        loss = self.loss_fct(confmap_pred, confmap_gts)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_id):
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']

        # Validation often uses batch_size=1 or small batch, usually not strictly sequential like train
        # But if using Record class, we might just want the prediction.
        # forward() handles unpacking and squeezing.

        confmap_pred = self.forward(ra_maps)

        # Handle GT shape if necessary. 
        # If val_loader is Many-to-One, GT might be (B, C, H, W).
        # If Many-to-Many, GT is (B, C, T, H, W).
        # If confmap_pred was squeezed to 4D, make sure GT matches.

        # Heuristic: align GT to pred
        if confmap_pred.dim() == 4 and confmap_gts.dim() == 5:
            # Take last frame of GT if we squeezed prediction (usually implies T=1 or evaluation on last frame)
            confmap_gts = confmap_gts[:, :, -1, :, :]

        loss = self.loss_fct(confmap_pred, confmap_gts)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
                 batch_size=self.batch_size)
        self.log('hp/val_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_id):
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']
        seq_name = batch['seq_names'][0]

        if confmap_gts is not None:
            save_dir = self.val_res_dir
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            save_dir = self.test_res_dir
            start_frame_name = image_paths[0][0][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, seq_name.upper() + ".txt")

        # Forward
        confmap_pred = self.forward(ra_maps)

        # Post-process requires CPU tensor (B=1, C, H, W)
        # forward() already squeezed T dim if T=1
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

    # ... (evaluate_rodnet_seq_, evaluate_rodnet_, on_before_batch_transfer, configure_optimizers 保持不变) ...
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
        self.logger.log_metrics({"AP/Overall": stats[0] * 100,
                                 "AR/Overall": stats[1] * 100})
        self.logger.log_metrics({"hp/AP": stats[0] * 100,
                                 "hp/AR": stats[1] * 100})

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
        scheduler = self.train_cfg['scheduler']
        if opti == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif opti == 'adam_reg':
            assert self.weight_decay is not None
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif opti == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif opti == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        else:
            raise ValueError
        if scheduler == 'exp':
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                'interval': 'epoch',
                'frequency': 10
            }
        elif scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif scheduler is None:
            return optimizer
        else:
            raise ValueError
        return [optimizer], [lr_scheduler]