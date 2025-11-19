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
        self.current_seq = None  # 用于记录当前正在处理的序列名

    def training_step(self, batch, batch_id):
        # Online 模式通常不用于训练，或者使用 BPTT
        # 如果必须保留，逻辑保持不变，但通常 Evaluation 不需要这个
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']
        total_loss = 0
        # 训练时每个 Batch 内部是一个序列片段，通常开头重置
        self.model.encoder.__init_hidden__()

        for t in range(ra_maps.shape[2]):
            confmap_pred = self.model(ra_maps[:, :, t])
            loss = self.loss_fct(confmap_pred, confmap_gts[:, :, t])
            total_loss += loss

        total_loss = total_loss / ra_maps.shape[2]
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        return total_loss

    def val_dataloader(self):
        # 必须保证 shuffle=False，否则状态会乱
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    def validation_step(self, batch, batch_id):
        """
        Modified validation step with state reset logic
        """
        ra_maps = batch['radar_data']
        confmap_gts = batch['anno']['confmaps']
        # 获取当前序列名称
        seq_name = batch['seq_names'][0]

        # === 关键修改：检测序列切换 ===
        if seq_name != self.current_seq:
            self.model.encoder.__init_hidden__()
            self.current_seq = seq_name
        # ============================

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."
        confmap_pred = self.forward(ra_maps[:, :, 0])

        loss = self.loss_fct(confmap_pred, confmap_gts[:, :, 0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Modified test step with robust state reset logic
        """
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']
        seq_name = batch['seq_names'][0]

        # === 关键修改：检测序列切换 ===
        # 这比 if frame_id == 0 更健壮，因为有些数据集切片未必从 0 开始，
        # 或者 DataLoader 可能会跨序列
        if seq_name != self.current_seq:
            self.model.encoder.__init_hidden__()
            self.current_seq = seq_name
        # ============================

        # 之前的 save_dir 逻辑
        if confmap_gts is not None:
            # confmap_gts = batch['anno']['confmaps'].float() # 这行好像多余，上面已经是 None 判断了
            save_dir = os.path.join(self.val_res_dir)
            # Frame ID parsing logic (保持原样或根据实际情况调整)
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            save_dir = os.path.join(self.test_res_dir)
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, seq_name.upper() + ".txt")

        assert ra_maps.shape[2] == 1 and ra_maps.shape[0] == 1, "Batch size and window size must be one for inference."
        confmap_pred = self.forward(ra_maps[:, :, 0])

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

    def on_validation_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None  # 重置序列追踪器

    def on_test_start(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None  # 重置序列追踪器

    def on_validation_end(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None

    def on_test_end(self):
        self.model.encoder.__init_hidden__()
        self.current_seq = None