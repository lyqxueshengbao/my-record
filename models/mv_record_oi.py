# models/mv_record_oi.py
# ----------------------------------------------------------------
import torch
import torch.nn as nn
from .record_oi import RECORD_OI
from .layers.ghost_module import GhostBottleneck  # <--- MODIFIED: 导入 GhostBottleneck
# from .layers.inverted_residual import InvertedResidual # <--- MODIFIED: 移除 InvertedResidual
from utils import get_norm_layer


class MV_RECORD_OI(nn.Module):
    def __init__(self, config):
        super(MV_RECORD_OI, self).__init__()
        self.config = config.model_config
        self.n_classes = self.config['n_classes']
        self.ra_encoder = RECORD_OI(config)
        self.rd_encoder = RECORD_OI(config)
        self.ad_encoder = RECORD_OI(config)
        self.norm_layer = get_norm_layer(self.config['norm'])

        self.conv_t1_ra = nn.ConvTranspose2d(self.ra_encoder.channels[7] * 3,
                                             self.ra_encoder.channels[5], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_ra = int(round(self.ra_encoder.channels[5] * 4))  # expand_ratio = 4
        self.conv9_ra = GhostBottleneck(
            in_chs=self.ra_encoder.channels[5],
            mid_chs=hidden_dim_ra,
            out_chs=self.ra_encoder.channels[5],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t2_ra = nn.ConvTranspose2d(self.ra_encoder.channels[5],
                                             self.ra_encoder.channels[3], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_ra_2 = int(round(self.ra_encoder.channels[3] * 4))  # expand_ratio = 4
        self.conv10_ra = GhostBottleneck(
            in_chs=self.ra_encoder.channels[3],
            mid_chs=hidden_dim_ra_2,
            out_chs=self.ra_encoder.channels[3],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t3_ra = nn.ConvTranspose2d(self.ra_encoder.channels[3],
                                             self.ra_encoder.channels[1], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_ra_3 = int(round(self.ra_encoder.channels[1] * 4))  # expand_ratio = 4
        self.conv11_ra = GhostBottleneck(
            in_chs=self.ra_encoder.channels[1],
            mid_chs=hidden_dim_ra_3,
            out_chs=self.ra_encoder.channels[1],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t4_ra = nn.ConvTranspose2d(self.ra_encoder.channels[1],
                                             self.ra_encoder.channels[0], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_ra_4 = int(round(self.ra_encoder.channels[0] * 1))  # expand_ratio = 1
        self.conv12_ra = GhostBottleneck(
            in_chs=self.ra_encoder.channels[0],
            mid_chs=hidden_dim_ra_4,
            out_chs=self.ra_encoder.channels[0],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.head_ra = nn.Conv2d(self.ra_encoder.channels[0], self.n_classes, 1)

        self.conv_t1_rd = nn.ConvTranspose2d(self.rd_encoder.channels[7] * 3,
                                             self.rd_encoder.channels[5], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_rd = int(round(self.rd_encoder.channels[5] * 4))  # expand_ratio = 4
        self.conv9_rd = GhostBottleneck(
            in_chs=self.rd_encoder.channels[5],
            mid_chs=hidden_dim_rd,
            out_chs=self.rd_encoder.channels[5],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t2_rd = nn.ConvTranspose2d(self.rd_encoder.channels[5],
                                             self.rd_encoder.channels[3], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_rd_2 = int(round(self.rd_encoder.channels[3] * 4))  # expand_ratio = 4
        self.conv10_rd = GhostBottleneck(
            in_chs=self.rd_encoder.channels[3],
            mid_chs=hidden_dim_rd_2,
            out_chs=self.rd_encoder.channels[3],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t3_rd = nn.ConvTranspose2d(self.rd_encoder.channels[3],
                                             self.rd_encoder.channels[1], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_rd_3 = int(round(self.rd_encoder.channels[1] * 4))  # expand_ratio = 4
        self.conv11_rd = GhostBottleneck(
            in_chs=self.rd_encoder.channels[1],
            mid_chs=hidden_dim_rd_3,
            out_chs=self.rd_encoder.channels[1],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.conv_t4_rd = nn.ConvTranspose2d(self.rd_encoder.channels[1],
                                             self.rd_encoder.channels[0], 2, 2)

        # <--- MODIFIED BLOCK START --->
        hidden_dim_rd_4 = int(round(self.rd_encoder.channels[0] * 1))  # expand_ratio = 1
        self.conv12_rd = GhostBottleneck(
            in_chs=self.rd_encoder.channels[0],
            mid_chs=hidden_dim_rd_4,
            out_chs=self.rd_encoder.channels[0],
            kernel_size=3,
            stride=1,
            norm_layer=self.norm_layer
        )
        # <--- MODIFIED BLOCK END --->

        self.head_rd = nn.Conv2d(self.rd_encoder.channels[0], self.n_classes, 1)

    def forward(self, x):
        ra, rd, ad = x
        b, f, c, h, w = ra.shape
        ra = ra.reshape(b * f, c, h, w)
        rd = rd.reshape(b * f, c, h, w)
        ad = ad.reshape(b * f, c, h, w)

        x_ra = self.ra_encoder.conv1(ra)
        x_ra = self.ra_encoder.conv2(x_ra)
        x_ra = self.ra_encoder.conv3(x_ra)
        x_ra_e_1 = self.ra_encoder.conv4(x_ra)
        h1_ra, c1_ra = self.ra_encoder.states[0]
        h1_ra, c1_ra = self.ra_encoder.lstm1(x_ra_e_1, (h1_ra, c1_ra))
        x_ra_e_1_lstm = h1_ra
        x_ra = self.ra_encoder.conv5(x_ra_e_1_lstm)
        x_ra_e_2 = self.ra_encoder.conv6(x_ra)
        h2_ra, c2_ra = self.ra_encoder.states[1]
        h2_ra, c2_ra = self.ra_encoder.lstm2(x_ra_e_2, (h2_ra, c2_ra))
        x_ra_e_2_lstm = h2_ra
        x_ra = self.ra_encoder.conv7(x_ra_e_2_lstm)
        x_ra = self.ra_encoder.conv8(x_ra)

        x_rd = self.rd_encoder.conv1(rd)
        x_rd = self.rd_encoder.conv2(x_rd)
        x_rd = self.rd_encoder.conv3(x_rd)
        x_rd_e_1 = self.rd_encoder.conv4(x_rd)
        h1_rd, c1_rd = self.rd_encoder.states[0]
        h1_rd, c1_rd = self.rd_encoder.lstm1(x_rd_e_1, (h1_rd, c1_rd))
        x_rd_e_1_lstm = h1_rd
        x_rd = self.rd_encoder.conv5(x_rd_e_1_lstm)
        x_rd_e_2 = self.rd_encoder.conv6(x_rd)
        h2_rd, c2_rd = self.rd_encoder.states[1]
        h2_rd, c2_rd = self.rd_encoder.lstm2(x_rd_e_2, (h2_rd, c2_rd))
        x_rd_e_2_lstm = h2_rd
        x_rd = self.rd_encoder.conv7(x_rd_e_2_lstm)
        x_rd = self.rd_encoder.conv8(x_rd)

        x_ad = self.ad_encoder.conv1(ad)
        x_ad = self.ad_encoder.conv2(x_ad)
        x_ad = self.ad_encoder.conv3(x_ad)
        x_ad_e_1 = self.ad_encoder.conv4(x_ad)
        h1_ad, c1_ad = self.ad_encoder.states[0]
        h1_ad, c1_ad = self.ad_encoder.lstm1(x_ad_e_1, (h1_ad, c1_ad))
        x_ad_e_1_lstm = h1_ad
        x_ad = self.ad_encoder.conv5(x_ad_e_1_lstm)
        x_ad_e_2 = self.ad_encoder.conv6(x_ad)
        h2_ad, c2_ad = self.ad_encoder.states[1]
        h2_ad, c2_ad = self.ad_encoder.lstm2(x_ad_e_2, (h2_ad, c2_ad))
        x_ad_e_2_lstm = h2_ad
        x_ad = self.ad_encoder.conv7(x_ad_e_2_lstm)
        x_ad = self.ad_encoder.conv8(x_ad)

        x = torch.cat((x_ra, x_rd, x_ad), dim=1)

        x_ra = self.conv_t1_ra(x)
        x_ra = x_ra + x_ra_e_2_lstm
        x_ra = self.conv9_ra(x_ra)
        x_ra = self.conv_t2_ra(x_ra)
        x_ra = x_ra + x_ra_e_1_lstm
        x_ra = self.conv10_ra(x_ra)
        x_ra = self.conv_t3_ra(x_ra)
        x_ra = self.conv11_ra(x_ra)
        x_ra = self.conv_t4_ra(x_ra)
        x_ra = self.conv12_ra(x_ra)

        x_rd = self.conv_t1_rd(x)
        x_rd = x_rd + x_rd_e_2_lstm
        x_rd = self.conv9_rd(x_rd)
        x_rd = self.conv_t2_rd(x_rd)
        x_rd = x_rd + x_rd_e_1_lstm
        x_rd = self.conv10_rd(x_rd)
        x_rd = self.conv_t3_rd(x_rd)
        x_rd = self.conv11_rd(x_rd)
        x_rd = self.conv_t4_rd(x_rd)
        x_rd = self.conv12_rd(x_rd)

        x_ra = self.head_ra(x_ra)
        x_rd = self.head_rd(x_rd)
        self.ra_encoder.states = ((h1_ra.detach(), c1_ra.detach()), (h2_ra.detach(), c2_ra.detach()))
        self.rd_encoder.states = ((h1_rd.detach(), c1_rd.detach()), (h2_rd.detach(), c2_rd.detach()))
        self.ad_encoder.states = ((h1_ad.detach(), c1_ad.detach()), (h2_ad.detach(), c2_ad.detach()))
        return x_ra, x_rd