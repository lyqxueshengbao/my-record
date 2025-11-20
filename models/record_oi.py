from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Entrainement en initialisant les états sur les k premières frames puis prédire les suivantes

Online learning LSTM biblio 

"""


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) - Online Inference Mode
        *** 调优版 (Hybrid Norm + Persistent States) ***
        """
        super(RecordOI, self).__init__()

        # 1. 保持与训练模型(Buffer模式)完全一致的结构
        # Stem 使用 BN，Recurrent 使用 LayerNorm/GroupNorm
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # 2. 内部状态持久化存储 (Memory)
        # 初始化为 None，表示还没有开始推理或刚被重置
        self.h_list = None
        self.c_list = None

    def reset_memory(self):
        """
        手动重置记忆。当开始一个新的视频流或切换场景时调用此函数。
        """
        self.h_list = None
        self.c_list = None

    def forward(self, x):
        """
        Forward pass for Online Inference
        @param x: (B, C, T, H, W)
                  通常 Online 模式下 T=1，但也支持 T>1 的小批量输入
        """
        B, C, T, H, W = x.shape

        # --- 1. Parallel Stem Execution (架构对齐) ---
        # 即使 T=1，保持这个结构也能确保与训练时的计算图一致
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        stem_features = self.encoder.forward_stem(x_reshaped)

        # Reshape back to (B, T, ...)
        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # --- 2. State Management (自动初始化与检查) ---
        # 如果状态为空，或者当前的 Batch Size 与缓存的状态不匹配（比如从单流变成了多流），则重置
        if self.h_list is None:
            self.encoder.__init_hidden__(batch_size=B, device=x.device)
            self.h_list = self.encoder.h_list
            self.c_list = self.encoder.c_list
        else:
            # 安全检查：防止输入的 Batch Size 突然变化导致维度不匹配报错
            # 假设 self.h_list[0] 是 (B, C, H, W)
            if self.h_list[0] is not None and self.h_list[0].shape[0] != B:
                # print("Warning: Batch size changed, resetting memory.")
                self.encoder.__init_hidden__(batch_size=B, device=x.device)
                self.h_list = self.encoder.h_list
                self.c_list = self.encoder.c_list

        # --- 3. Recurrent Loop (更新并保存状态) ---
        st_features_backbone = None
        st_features_lstm2 = None
        st_features_lstm1 = None

        for t in range(T):
            x_t = recurrent_input[:, t, ...]

            # 使用上一时刻的 self.h_list, self.c_list 进行更新
            # 并将更新后的状态直接写回 self.h_list, self.c_list
            (st_features_backbone,
             st_features_lstm2,
             st_features_lstm1), self.h_list, self.c_list = self.encoder.forward_recurrent_step(
                x_t, self.h_list, self.c_list
            )

        # --- 4. Decoder ---
        # 解码最后时刻的特征
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)

