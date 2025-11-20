# --- START OF FILE cruw_trainer_oi.py ---

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
        """
        # Get data
        ra_maps = batch['radar_data']  # N, C, T, H, W
        confmap_gts = batch['anno']['confmaps']

        total_loss = 0

        # 如果是训练模式，每个Batch开始前重置记忆
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        else:
            self.model.encoder.__init_hidden__()

        for t in range(ra_maps.shape[2]):
            # 修改点 1: 增加维度，确保输入为 (B, C, 1, H, W)
            # ra_maps[:, :, t] 是 (B, C, H, W) -> unsqueeze(2) -> (B, C, 1, H, W)
            input_frame = ra_maps[:, :, t].unsqueeze(2)

            confmap_pred = self.model(input_frame)

            # 计算 Loss
            loss = self.loss_fct(confmap_pred, confmap_gts[:, :, t])
            total_loss += loss

        total_loss = total_loss / ra_maps.shape[2]
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', total_loss, on_epoch=True)

        return total_loss

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step
        """
        ra_maps = batch['radar_data']  # N, C, T, H, W
        confmap_gts = batch['anno']['confmaps']

        # 修改点 2: 确保输入维度匹配 (B, C, 1, H, W)
        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # ra_maps[:, :, 0] 会变成 4D，我们需要保留 T 维度，或者手动 unsqueeze
        input_frame = ra_maps[:, :, 0:1]  # 切片保持维度 (B, C, 1, H, W)

        confmap_pred = self.forward(input_frame)

        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step
        """
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
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            start_frame_name = image_paths[0][0][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)
        if frame_id % 16 == 0:
            self.model.reset_memory()
        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 修改点 3: 确保输入维度匹配 (B, C, 1, H, W)
        input_frame = ra_maps[:, :, 0:1]  # 使用切片 0:1 保持维度
        confmap_pred = self.forward(input_frame)

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

    def evaluate_rodnet_(self):
        # ... (保持原样) ...
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        out_eval = accumulate(self.evalImgs_all, self.n_frames_all, ols_thrs, rec_thrs, self.cruw_dataset_obj,
                              log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)
        self.logger.log_metrics({"AP/Overall": stats[0] * 100,
                                 "AR/Overall": stats[1] * 100})

        self.logger.log_metrics({"hp/AP": stats[0] * 100,
                                 "hp/AR": stats[1] * 100})

    # 修改点 4: 使用 reset_memory 替换 __init_hidden__
    # 兼容性写法：如果模型有 reset_memory 就用，没有就回退到旧方法
    def _reset_model_memory(self):
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        else:
            # Fallback for old models
            self.model.encoder.__init_hidden__()

    def on_validation_start(self):
        self._reset_model_memory()

    def on_test_start(self):
        # 重要：eval_cruw.py 会对每个 Sequence 调用一次 trainer.test()
        # 所以这里就是每个视频开始的地方，必须重置记忆！
        self._reset_model_memory()

    def on_validation_end(self):
        self._reset_model_memory()

    def on_test_end(self):
        self._reset_model_memory()