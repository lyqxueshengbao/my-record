import torch
from torch import nn
from .record import RecordEncoder, RecordDecoder


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        super(RecordOI, self).__init__()

        # 1. Encoder 配置 (保持 bn 以加载权重)
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        # 2. Decoder 配置
        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # 3. 初始化状态 (修复 Bug)
        self.encoder.__init_hidden__()

    def forward(self, x):
        """
        Forward pass for Online Inference (Single Frame)
        x shape: (B, C, H, W)
        """
        # === 关键修改开始 ===
        # 不再调用 self.encoder(x)，而是分步调用以适配 Mixed Norm Encoder

        # 1. 执行 Stem 部分 (BN)
        # x: (B, C, H, W)
        stem_features = self.encoder.forward_stem(x)

        # 2. 执行 Recurrent 部分 (GN + LSTM)
        # 使用 encoder 内部保存的 h_list 和 c_list
        # 如果是刚开始 (None)，则初始化
        if self.encoder.h_list[0] is None:
            self.encoder.__init_hidden__()

        (st_features_backbone,
         st_features_lstm2,
         st_features_lstm1), h_list, c_list = self.encoder.forward_recurrent_step(
            stem_features,
            self.encoder.h_list,
            self.encoder.c_list
        )

        # 3. 更新状态
        # 注意：这里将更新后的状态存回 self.encoder，供下一帧使用
        self.encoder.h_list = h_list
        self.encoder.c_list = c_list
        # === 关键修改结束 ===

        # 4. Decoder
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)

    # 将此方法添加到 models/record_oi.py 的 RecordOI 类中
    def train(self, mode=True):
        """
        重写 train 方法：
        在进入 Training 模式时，强制将 Encoder Stem 部分的 BN 层保持在 Eval 模式。
        这能防止因 Batch Size 过小导致的统计量崩坏，保护预训练权重。
        """
        super().train(mode)  # 先让全网进入 mode 指定的状态

        if mode:  # 只有在切换到 Training 状态时才需要干预
            # 遍历 Encoder Stem (BN部分) 的所有层
            for m in self.encoder.stem.modules():
                # 如果是 BN 层，强行按住它的头，不让它动
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

            # (可选) 如果你想更彻底一点，连 BN 的 γ 和 β 参数都不让学，可以加上这句：
            # for param in self.encoder.stem.parameters():
            #     param.requires_grad = False

            # 打印一次提示，确保你看到了它生效
            # print("Info: Stem BN layers frozen in EVAL mode for Online Training.")
    def reset_hidden(self):
        self.encoder.__init_hidden__()