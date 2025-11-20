import torch
import torch.nn as nn


# 请确保正确导入了 RecordEncoder 和 RecordDecoder
# from models.record_encoder import RecordEncoder 
# from models.record_decoder import RecordDecoder

class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) - Online Inference Mode
        *** 调优版 (Hybrid Norm + Persistent States) - FIX VERSION ***
        """
        super(RecordOI, self).__init__()

        # 1. 保持与训练模型(Buffer模式)完全一致的结构
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # 2. 内部状态持久化存储
        self.h_list = None
        self.c_list = None

    def reset_memory(self):
        """
        手动重置记忆。
        """
        self.h_list = None
        self.c_list = None
        # 同时调用 encoder 的 reset 以防它内部也有状态
        if hasattr(self.encoder, '__init_hidden__'):
            self.encoder.__init_hidden__()

    def forward(self, x):
        """
        Forward pass for Online Inference
        @param x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # --- 1. Parallel Stem Execution ---
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        stem_features = self.encoder.forward_stem(x_reshaped)

        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # --- 2. State Management (修复了这里) ---
        # 情况 A: 还没有初始化
        if self.h_list is None:
            # remove args: batch_size=B, device=x.device
            self.encoder.__init_hidden__()
            self.h_list = self.encoder.h_list
            self.c_list = self.encoder.c_list

        # 情况 B: 已经初始化，但 Batch Size 变了 (例如从 1 变成 4)
        # 我们需要检查 h_list 中的第一个非 None 元素来确认形状
        elif self.h_list[0] is not None:
            # 假设 hidden state 是 (Batch, Channel, H, W)
            current_state_batch = self.h_list[0].shape[0]
            if current_state_batch != B:
                # print(f"Batch size changed from {current_state_batch} to {B}, resetting memory.")
                # remove args here too
                self.encoder.__init_hidden__()
                self.h_list = self.encoder.h_list
                self.c_list = self.encoder.c_list

        # --- 3. Recurrent Loop ---
        st_features_backbone = None
        st_features_lstm2 = None
        st_features_lstm1 = None

        for t in range(T):
            x_t = recurrent_input[:, t, ...]

            # RecordEncoder 会自动处理 h_list 为 [None, None] 的情况，
            # 在内部根据 x_t 的形状生成全 0 张量。
            (st_features_backbone,
             st_features_lstm2,
             st_features_lstm1), self.h_list, self.c_list = self.encoder.forward_recurrent_step(
                x_t, self.h_list, self.c_list
            )

        # --- 4. Decoder ---
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)