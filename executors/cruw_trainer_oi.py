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
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        # Get data
        ra_maps = batch['radar_data']  # N, C, T, H, W
        confmap_gts = batch['anno']['confmaps']  # N, 1, T, H, W
        image_paths = batch['image_paths']

        total_loss = 0
        T = ra_maps.shape[2]

        for t in range(T):
            if t == 0:
                self.model.encoder.__init_hidden__()

            # 🔥 关键修改：保持5维
            confmap_pred = self.model(ra_maps[:, :, t:t + 1])  # (N, C, 1, H, W)
            confmap_gt = confmap_gts[:, :, t:t + 1]  # (N, 1, 1, H, W)

            # 如果损失函数需要4维，在这里squeeze
            loss = self.loss_fct(confmap_pred.squeeze(2), confmap_gt.squeeze(2))
            # 或者如果损失函数支持5维：
            # loss = self.loss_fct(confmap_pred, confmap_gt)

            total_loss += loss

        # Loss averaging over frames
        total_loss = total_loss / T
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', total_loss, on_epoch=True)

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
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        # Get data
        ra_maps = batch['radar_data']  # N, C, T, H, W
        confmap_gts = batch['anno']['confmaps']  # N, 1, T, H, W
        image_paths = batch['image_paths']
        obj_infos = batch['anno']['obj_infos']

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 🔥 关键修改：保持5维
        confmap_pred = self.forward(ra_maps[:, :, 0:1])  # (1, C, 1, H, W)
        confmap_gt = confmap_gts[:, :, 0:1]  # (1, 1, 1, H, W)

        # 如果损失函数需要4维
        loss = self.loss_fct(confmap_pred.squeeze(2), confmap_gt.squeeze(2))
        # 或者如果损失函数支持5维：
        # loss = self.loss_fct(confmap_pred, confmap_gt)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        ra_maps = batch['radar_data']  # N, C, T, H, W
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

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 🔥 关键修改：保持5维
        confmap_pred = self.forward(ra_maps[:, :, 0:1])  # (1, C, 1, H, W)
        confmap_pred = confmap_pred.squeeze(2)  # (1, C, H, W) - 用于后处理

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

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

    def on_validation_start(self):
        self.model.encoder.__init_hidden__()

    def on_test_start(self):
        self.model.encoder.__init_hidden__()

    def on_validation_end(self):
        self.model.encoder.__init_hidden__()

    def on_test_end(self):
        self.model.encoder.__init_hidden__()
