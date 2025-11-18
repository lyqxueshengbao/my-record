import torch
from torch import nn
from .record import RecordEncoder, RecordDecoder


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        super(RecordOI, self).__init__()

        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # ✅ 调用 __init_hidden__() 只是设置为 None
        # 实际初始化会在 BottleneckLSTM 内部自动完成
        self.encoder.__init_hidden__()

    def load_buffer_weights_and_freeze_bn(self, checkpoint_path):
        """加载 Buffer 权重并完全冻结 BN"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        self.freeze_bn_layers()
        print("✅ Loaded weights and frozen BN layers")

    def freeze_bn_layers(self):
        """完全冻结所有 BN 层（在 stem 中）"""
        for module in self.encoder.stem.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                module.track_running_stats = False
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """重写 train 方法，确保 BN 始终冻结"""
        super().train(mode)
        if mode:
            self.freeze_bn_layers()
        return self

    def forward(self, x):
        """
        Forward pass for Online Inference (Single Frame)
        @param x: input tensor with shape (B, C, H, W)
        @return: confidence map with shape (B, n_class, H, W)
        """
        # 1. Stem 部分（使用 BatchNorm，需要冻结）
        stem_features = self.encoder.forward_stem(x)
        # stem_features shape: (B, C_feat, H_feat, W_feat)

        # 2. Recurrent 部分（使用 LayerNorm/GroupNorm）
        # BottleneckLSTM 会自动处理 None 状态的初始化
        (st_features_backbone, st_features_lstm2, st_features_lstm1), h_list, c_list = \
            self.encoder.forward_recurrent_step(
                stem_features,
                self.encoder.h_list,
                self.encoder.c_list
            )

        # 3. 更新隐藏状态（用于下一帧）
        self.encoder.h_list = h_list
        self.encoder.c_list = c_list

        # 4. Decoder
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)

        return self.sigmoid(confmap_pred)

    def reset_hidden(self):
        """
        重置隐藏状态
        用于：
        1. 开始处理新的视频序列
        2. 切换到不相关的帧流
        """
        # 调用 encoder 的 __init_hidden__()，设置为 None
        self.encoder.__init_hidden__()
