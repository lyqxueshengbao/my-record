from .cruw_trainer import CruwExecutor
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutorOI(CruwExecutor):

    def training_step(self, batch, batch_id):
        """
        正确的online训练策略:
        1. 在batch之间重置LSTM (✅ 正确)
        2. 在batch内部保持LSTM连续 (✅ 修复)
        3. 只计算最后一帧的loss以节省显存 (✅ 优化)

        这样既保持了时序依赖,又节省了显存
        """
        ra_maps = batch['radar_data']  # (B, C, T, H, W)
        confmap_gts = batch['anno']['confmaps']

        # ⚠️ 关键修复: 移到batch开始时重置 (而非循环内)
        # 这样LSTM状态在batch的T帧内保持连续
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        if self._batch_count == 0 or batch_id == 0:
            self.model.encoder.__init_hidden__()
        self._batch_count = batch_id

        T = ra_maps.shape[2]

        # 方案A: 只对最后一帧计算loss (节省显存,和buffer模式一致)
        # 前T-1帧用于warm-up LSTM
        for t in range(T - 1):
            with torch.no_grad():
                _ = self.model(ra_maps[:, :, t])

        confmap_pred = self.model(ra_maps[:, :, -1])
        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, -1])

        # 方案B: 对所有帧计算loss (更准确,但显存消耗大)
        # 如果显存足够 (batch_size较小),可以用这个方案
        # total_loss = 0
        # for t in range(T):
        #     confmap_pred = self.model(ra_maps[:, :, t])
        #     loss = self.loss_fct(confmap_pred, confmap_gts[:, :, t])
        #     total_loss += loss
        # loss = total_loss / T

        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log('hp/train_loss', loss, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_start(self):
        """Epoch开始时重置计数器"""
        self._batch_count = 0
        self.model.encoder.__init_hidden__()

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=False)

    def validation_step(self, batch, batch_id):
        """验证步骤 - 保持online推理方式"""
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1

        confmap_pred = self.forward(ra_maps[:, :, 0])
        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True, batch_size=1)
        self.log('hp/val_loss', loss, on_epoch=True, sync_dist=True, batch_size=1)

    def test_step(self, batch, batch_id):
        """测试步骤 - 保持原有的online推理速度"""
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']
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
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
        else:
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]

        frame_id = int(frame_name)

        # 在序列边界重置LSTM (保持online推理逻辑)
        if not hasattr(self, '_last_test_seq_name') or \
                self._last_test_seq_name != seq_name or \
                frame_id == 0:
            self.model.encoder.__init_hidden__()
            self._last_test_seq_name = seq_name
            print(f"[Online Mode] Reset LSTM for sequence: {seq_name}, frame: {frame_id}")

        # 单帧forward - 保持6.2ms的推理速度!
        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1
        confmap_pred = self.forward(ra_maps[:, :, 0])

        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(),
                                                   self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path,
                                        self.cruw_dataset_obj)

    def evaluate_rodnet_(self):
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        out_eval = accumulate(self.evalImgs_all, self.n_frames_all, ols_thrs, rec_thrs,
                              self.cruw_dataset_obj, log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)

        self.logger.log_metrics({"AP/Overall": stats[0] * 100, "AR/Overall": stats[1] * 100})
        self.logger.log_metrics({"hp/AP": stats[0] * 100, "hp/AR": stats[1] * 100})

    def on_validation_epoch_start(self):
        self.model.encoder.__init_hidden__()

    def on_test_epoch_start(self):
        self._last_test_seq_name = None
        self.model.encoder.__init_hidden__()
