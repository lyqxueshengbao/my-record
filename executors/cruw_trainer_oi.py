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
        ra_maps = batch['radar_data']  # N, C, T, H, W (T=32)
        confmap_gts = batch['anno']['confmaps']  # N, Class, T, H, W
        image_paths = batch['image_paths']
        total_loss = 0

        # <--- MODIFICATION START ---
        # 关键修复：
        # 我们在 *批次(batch)* 开始时重置隐藏状态，而不是在 *循环(loop)* 内部。
        # 这确保了模型在 32 帧的窗口内保持记忆连续性，
        # 从而正确学习时序依赖关系。
        self.model.encoder.__init_hidden__()
        # <--- MODIFICATION END ---

        for t in range(ra_maps.shape[2]):  # 循环 32 帧
            # <--- MODIFICATION START ---
            # 删除了原有的 'if t == 0: self.model.encoder.__init_hidden__()'
            # <--- MODIFICATION END ---

            confmap_pred = self.model(ra_maps[:, :, t])
            loss = self.loss_fct(confmap_pred, confmap_gts[:, :, t])
            total_loss += loss

        # TODO: loss depending on the frames (如我们之前讨论的，这个TODO是作者想做加权损失的标记)
        total_loss = total_loss / ra_maps.shape[2]
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
        # (此函数无需修改)
        # Get data
        ra_maps = batch['radar_data']  # N, C, T, H, W (T=1)
        confmap_gts = batch['anno']['confmaps']  # N, Class, T, H, W
        image_paths = batch['image_paths']
        obj_infos = batch['anno']['obj_infos']

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 'forward' 会调用 self.model()，它会连续使用并更新隐藏状态
        confmap_pred = self.forward(ra_maps[:, :, 0])

        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        # (此函数无需修改)
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

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 'forward' 会调用 self.model()，它会连续使用并更新隐藏状态
        confmap_pred = self.forward(ra_maps[:, :, 0])

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
        # (此函数无需修改)
        # 在验证开始时（即处理第一个序列之前）重置状态。
        self.model.encoder.__init_hidden__()

    def on_test_start(self):
        # (此函数无需修改)
        # 在测试开始时（即处理第一个序列之前）重置状态。
        # 评估脚本 'evaluation/eval_cruw.py'
        # 会为 *每个* 序列调用一次 'trainer.test()'，
        # 因此这个钩子会在每个序列开始时被正确调用。
        self.model.encoder.__init_hidden__()

    def on_validation_end(self):
        # (此函数无需修改)
        self.model.encoder.__init_hidden__()

    def on_test_end(self):
        # (此函数无需修改)
        self.model.encoder.__init_hidden__()