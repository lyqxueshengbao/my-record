from .cruw_trainer import CruwExecutor
import os
import numpy as np
from torch.utils.data import DataLoader
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutorOI(CruwExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用于记录当前正在处理的序列名，检测视频切换
        self.current_seq = None

    def training_step(self, batch, batch_id):
        """
        Online 模式的训练步（虽然通常只用于评估，但保留以防万一）
        """
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']

        # 训练时每个 Batch 内部视为一个序列片段，开头重置状态
        # 注意：真正的 Online 训练建议使用 TBPTT (如 cruw_trainer.py 所示)，
        # 这里保留原始逻辑供参考
        self.model.encoder.__init_hidden__()

        total_loss = 0
        for t in range(ra_maps.shape[2]):
            # Forward
            out = self.model(ra_maps[:, :, t])

            # 兼容性处理：如果返回 (pred, h, c)，只取 pred
            if isinstance(out, tuple):
                confmap_pred = out[0]
            else:
                confmap_pred = out

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
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']

        # === 核心修改：检查序列是否切换 ===
        if 'seq_names' in batch:
            # batch['seq_names'] 是一个列表，batch_size=1 时取第0个
            seq_name = batch['seq_names'][0]
            if seq_name != self.current_seq:
                # 换视频了，重置 LSTM 状态
                self.model.encoder.__init_hidden__()
                self.current_seq = seq_name
        # ================================

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # 调用父类的 forward (已经修改为支持解包)
        confmap_pred = self.forward(ra_maps[:, :, 0])

        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']

        # 获取序列名
        seq_name = batch['seq_names'][0]

        # === 核心修改：检查序列是否切换 ===
        if seq_name != self.current_seq:
            # print(f"Switching sequence: {self.current_seq} -> {seq_name}, Resetting Hidden State!")
            self.model.encoder.__init_hidden__()
            self.current_seq = seq_name
        # ================================

        if confmap_gts is not None:
            save_dir = os.path.join(self.val_res_dir)
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            save_dir = os.path.join(self.test_res_dir)
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, seq_name.upper() + ".txt")

        # =========================================================
        # 这里的 frame_id % 12 已经被移除，
        # 因为模型已经通过 TBPTT 学会了长时记忆，且有序列切换重置逻辑兜底
        # =========================================================

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."

        # Forward
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

    # --- Hook 函数：确保每个阶段开始时状态清空 ---

    def on_validation_epoch_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None

    def on_test_epoch_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None

    # 兼容旧版 PL 的钩子命名（可选）
    def on_validation_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None

    def on_test_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None