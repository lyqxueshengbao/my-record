# models/record.py
# ----------------------------------------------------------------
import torch.nn as nn
from .layers.bottleneck_lstm import BottleneckLSTM
from .layers.ghost_module import GhostBottleneck  # <--- MODIFIED: 导入 GhostBottleneck
# from .layers.inverted_residual import InvertedResidual # <--- MODIFIED: 移除 InvertedResidual
from utils import get_norm_layer


class RECORD(nn.Module):

    def __init__(self, config):
        super(RECORD, self).__init__()
        self.config = config.model_config
        self.in_channels = self.config['in_channels']
        self.channels = self.config['channels']
        self.n_classes = self.config['n_classes']
        self.n_frames = self.config['n_frames']
        self.norm_layer = get_norm_layer(self.config['norm'])

        # <--- MODIFIED BLOCK START --->
        # conv1
        # 计算隐藏维度 (mid_chs)，与原 InvertedResidual 保持一致
        hidden_dim_0 = int(round(self.in_channels * 1))
        self.conv1 = GhostBottleneck(
            in_chs=self.in_channels,
            mid_chs=hidden_dim_0,
            out_chs=self.channels[0],
            kernel_size=3,
            stride=2,
            norm_layer=self.norm_layer
        )

        # conv2
        hidden_dim_1 = int(round(self.channels[0] * 4))  # expand_ratio = 4
        self.conv2 = GhostBottleneck(
            in_chs=self.channels[0],
            mid_chs=hidden_dim_1,
            out_chs=self.channels[1],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )

        # conv3
        hidden_dim_2 = int(round(self.channels[1] * 4))  # expand_ratio = 4
        self.conv3 = GhostBottleneck(
            in_chs=self.channels[1],
            mid_chs=hidden_dim_2,
            out_chs=self.channels[2],
            kernel_size=3,
            stride=2,
            norm_layer=self.norm_layer
        )

        # conv4
        hidden_dim_3 = int(round(self.channels[2] * 4))  # expand_ratio = 4
        self.conv4 = GhostBottleneck(
            in_chs=self.channels[2],
            mid_chs=hidden_dim_3,
            out_chs=self.channels[3],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )

        # conv5
        hidden_dim_4 = int(round(self.channels[3] * 4))  # expand_ratio = 4
        self.conv5 = GhostBottleneck(
            in_chs=self.channels[3],
            mid_chs=hidden_dim_4,
            out_chs=self.channels[4],
            kernel_size=3,
            stride=2,
            norm_layer=self.norm_layer
        )

        # conv6
        hidden_dim_5 = int(round(self.channels[4] * 4))  # expand_ratio = 4
        self.conv6 = GhostBottleneck(
            in_chs=self.channels[4],
            mid_chs=hidden_dim_5,
            out_chs=self.channels[5],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )

        # conv7
        hidden_dim_6 = int(round(self.channels[5] * 4))  # expand_ratio = 4
        self.conv7 = GhostBottleneck(
            in_chs=self.channels[5],
            mid_chs=hidden_dim_6,
            out_chs=self.channels[6],
            kernel_size=3,
            stride=2,
            norm_layer=self.norm_layer
        )

        # conv8
        hidden_dim_7 = int(round(self.channels[6] * 4))  # expand_ratio = 4
        self.conv8 = GhostBottleneck(
            in_chs=self.channels[6],
            mid_chs=hidden_dim_7,
            out_chs=self.channels[7],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.lstm1 = BottleneckLSTM(
            self.channels[3], self.channels[3],
            kernel_size=(3, 3),
            norm_layer=self.norm_layer
        )
        self.lstm2 = BottleneckLSTM(
            self.channels[5], self.channels[5],
            kernel_size=(3, 3),
            norm_layer=self.norm_layer
        )

        self.conv_t1 = nn.ConvTranspose2d(
            self.channels[7], self.channels[5],
            kernel_size=2, stride=2,
        )

        # <--- MODIFIED BLOCK START --->
        hidden_dim_8 = int(round(self.channels[5] * 4))  # expand_ratio = 4
        self.conv9 = GhostBottleneck(
            in_chs=self.channels[5],
            mid_chs=hidden_dim_8,
            out_chs=self.channels[5],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t2 = nn.ConvTranspose2d(
            self.channels[5], self.channels[3],
            kernel_size=2, stride=2,
        )

        # <--- MODIFIED BLOCK START --->
        hidden_dim_9 = int(round(self.channels[3] * 4))  # expand_ratio = 4
        self.conv10 = GhostBottleneck(
            in_chs=self.channels[3],
            mid_chs=hidden_dim_9,
            out_chs=self.channels[3],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t3 = nn.ConvTranspose2d(
            self.channels[3], self.channels[1],
            kernel_size=2, stride=2,
        )

        # <--- MODIFIED BLOCK START --->
        hidden_dim_10 = int(round(self.channels[1] * 4))  # expand_ratio = 4
        self.conv11 = GhostBottleneck(
            in_chs=self.channels[1],
            mid_chs=hidden_dim_10,
            out_chs=self.channels[1],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t4 = nn.ConvTranspose2d(
            self.channels[1], self.channels[0],
            kernel_size=2, stride=2,
        )

        # <--- MODIFIED BLOCK START --->
        hidden_dim_11 = int(round(self.channels[0] * 1))  # expand_ratio = 1
        self.conv12 = GhostBottleneck(
            in_chs=self.channels[0],
            mid_chs=hidden_dim_11,
            out_chs=self.channels[0],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.head = nn.Conv2d(self.channels[0], self.n_classes, kernel_size=1)

    def forward(self, x, states):
        # states
        (h1, c1), (h2, c2) = states

        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_e_1 = self.conv4(x)
        h1, c1 = self.lstm1(x_e_1, (h1, c1))
        x_e_1_lstm = h1
        x = self.conv5(x_e_1_lstm)
        x_e_2 = self.conv6(x)
        h2, c2 = self.lstm2(x_e_2, (h2, c2))
        x_e_2_lstm = h2
        x = self.conv7(x_e_2_lstm)
        x = self.conv8(x)

        # decoder
        x = self.conv_t1(x)
        x = x + x_e_2_lstm
        x = self.conv9(x)
        x = self.conv_t2(x)
        x = x + x_e_1_lstm
        x = self.conv10(x)
        x = self.conv_t3(x)
        x = self.conv11(x)
        x = self.conv_t4(x)
        x = self.conv12(x)

        # head
        x = self.head(x)
        return x, ((h1, c1), (h2, c2))