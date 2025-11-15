from .cruw_trainer import CruwExecutor
import os
import numpy as np
from torch.utils.data import DataLoader
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutorOI(CruwExecutor):

    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        *** 修改为支持 Hybrid Norm 的版本 ***

        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        # Get data
        ra_maps = batch['radar_data']  # (B, C, T, H, W)
        confmap_gts = batch['anno']['confmaps']  # (B, n_class, T, H, W)
        image_paths = batch['image_paths']

        B, C, T, H, W = ra_maps.shape

        # 初始化隐藏状态
        self.model.encoder.__init_hidden__()

        total_loss = 0

        # 逐时间步处理 (符合 Online 推理逻辑)
        for t in range(T):
            # 获取当前时间步的输入: (B, C, H, W)
            x_t = ra_maps[:, :, t]

            # Forward pass (会自动更新隐藏状态)
            confmap_pred = self.model(x_t)

            # 计算损失
            loss = self.loss_fct(confmap_pred, confmap_gts[:, :, t])
            total_loss += loss

        # 平均损失
        total_loss = total_loss / T

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True,
                 sync_dist=True, batch_size=self.batch_size)
        self.log('hp/train_loss', total_loss, on_epoch=True, sync_dist=True,
                 batch_size=self.batch_size)

        return total_loss

    def val_dataloader(self):
        """
        Define PyTorch validation dataloader
        @return: validation dataloader for ROD2021 dataset
        """
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        *** 修改为支持 Hybrid Norm 的版本 ***

        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        # Get data
        ra_maps = batch['radar_data']  # (B, C, T, H, W)
        confmap_gts = batch['anno']['confmaps']
        image_paths = batch['image_paths']
        obj_infos = batch['anno']['obj_infos']
        # vvvvvvvv   添加下面的 "重置状态" 逻辑   vvvvvvvvvv
        seq_name = batch['seq_names'][0]
        if seq_name != self.current_val_seq:
            self.model.encoder.__init_hidden__()
            self.current_val_seq = seq_name
        # ^^^^^^^^^^   添加上面的 "重置状态" 逻辑   ^^^^^^^^^^
        # Online inference: 每次处理单个时间步
        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, \
            "Batch size and window size must be one for online inference."

        # 提取单个时间步: (1, C, 1, H, W) -> (1, C, H, W)
        x_t = ra_maps[:, :, 0]

        # Forward (会自动使用当前隐藏状态)
        confmap_pred = self.forward(x_t)

        # 计算损失
        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False,
                 logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log('hp/val_loss', loss, on_epoch=True, sync_dist=True,
                 batch_size=self.batch_size)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        *** 修改为支持 Hybrid Norm 的版本 ***

        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']

        # Get seq name to write results
        seq_name = batch['seq_names'][0]
        # vvvvvvvv   添加下面的 "重置状态" 逻辑   vvvvvvvvvv
        if seq_name != self.current_test_seq:
            self.model.encoder.__init_hidden__()
            self.current_test_seq = seq_name
        # ^^^^^^^^^^   添加上面的 "重置状态" 逻辑   ^^^^^^^^^^
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

        # Online inference: 每次处理单个时间步
        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, \
            "Batch size and window size must be one for online inference."

        # 提取单个时间步: (1, C, 1, H, W) -> (1, C, H, W)
        x_t = ra_maps[:, :, 0]

        # Forward (会自动使用当前隐藏状态)
        confmap_pred = self.forward(x_t)

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(),
                                                   self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path,
                                        self.cruw_dataset_obj)

    def evaluate_rodnet_(self):
        """
        Evaluate RODNET performance
        """
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1),
                                         endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1),
                                         endpoint=True), decimals=2)
        out_eval = accumulate(self.evalImgs_all, self.n_frames_all, ols_thrs, rec_thrs,
                              self.cruw_dataset_obj, log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)

        self.logger.log_metrics({"AP/Overall": stats[0] * 100,
                                 "AR/Overall": stats[1] * 100})
        self.logger.log_metrics({"hp/AP": stats[0] * 100,
                                 "hp/AR": stats[1] * 100})

    def on_validation_start(self):
        """在验证开始时初始化隐藏状态"""
        self.model.encoder.__init_hidden__()

    def on_test_start(self):
        """在测试开始时初始化隐藏状态"""
        self.model.encoder.__init_hidden__()

    def on_validation_end(self):
        """在验证结束时重置隐藏状态"""
        self.model.encoder.__init_hidden__()

    def on_test_end(self):
        """在测试结束时重置隐藏状态"""
        self.model.encoder.__init_hidden__()