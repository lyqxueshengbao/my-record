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

        # ✅ 正确方式：初始化为 None，让 forward 自动处理
        self.reset_hidden()

    def load_buffer_weights_and_freeze_bn(self, checkpoint_path):
        """加载 Buffer 权重并完全冻结 BN"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        self.freeze_bn_layers()
        print("✅ Loaded weights and frozen BN layers")

    def freeze_bn_layers(self):
        """完全冻结所有 BN 层"""
        for module in self.encoder.stem.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                module.track_running_stats = False  # 停止更新统计量
                for param in module.parameters():
                    param.requires_grad = False  # 冻结参数

    def train(self, mode=True):
        """重写 train 方法，确保 BN 始终冻结"""
        super().train(mode)
        if mode:
            self.freeze_bn_layers()
        return self

    def forward(self, x):
        """
        Forward pass for Online Inference (Single Frame)
        x shape: (B, C, H, W)
        """
        # 1. Stem 部分（BN）
        stem_features = self.encoder.forward_stem(x)
        # stem_features shape: (B, C', H', W')

        # 2. 检查并初始化隐藏状态
        B, C, H, W = stem_features.shape

        # ✅ 正确的初始化逻辑
        if self.encoder.h_list[0] is None or self.encoder.h_list[0].shape[0] != B:
            # 根据实际的 stem_features 形状初始化
            self.encoder.__init_hidden__(batch_size=B, spatial_size=(H, W))

        # 3. Recurrent 部分（GN + LSTM）
        (st_features_backbone, st_features_lstm2, st_features_lstm1), h_list, c_list = \
            self.encoder.forward_recurrent_step(
                stem_features,
                self.encoder.h_list,
                self.encoder.c_list
            )

        # 4. 更新隐藏状态
        self.encoder.h_list = h_list
        self.encoder.c_list = c_list

        # 5. Decoder
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)

    def reset_hidden(self):
        """
        重置隐藏状态为未初始化状态
        用于：
        1. 开始处理新的视频序列
        2. 切换到不相关的帧流
        """
        # ✅ 正确方式：设置为 None，而不是用默认值初始化
        self.encoder.h_list = [None, None]  # 假设有两个 LSTM 层
        self.encoder.c_list = [None, None]
