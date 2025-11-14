#
# 完整文件: datasets/cruw/cruw_dataset.py
# (已替换为 CutMix)
#
import os
import random
import pickle
import numpy as np
import time
from .parse_pkl import list_pkl_filenames_from_prepared
from .transforms import random_apply, sequence_mixup, spatial_mixup

from torch.utils import data
import torch


class ROD2021Dataset(data.Dataset):
    def __init__(self, data_dir, dataset, config_dict, split, is_random_chirp=False, subset=None, all_confmaps=False):
        """
        Dataset for the ROD2021 dataset. Modified from: https://github.com/yizhou-wang/RODNet
        """
        # parameters settings
        self.data_dir = data_dir
        self.dataset = dataset
        self.config_dict = config_dict
        self.n_class = dataset.object_cfg.n_class
        self.win_size = config_dict['train_cfg']['win_size']
        self.all_confmaps = all_confmaps
        self.model_name = config_dict['model_cfg']['name']
        self.aug_dict = config_dict['train_cfg']['aug']

        # vvvvvvvvvvvv 【修改】 vvvvvvvvvvvv
        # CutMix configuration
        self.use_cutmix = config_dict['train_cfg'].get('use_cutmix', False)
        self.cutmix_alpha = config_dict['train_cfg'].get('cutmix_alpha', 1.0)  # CutMix 默认 alpha 为 1.0
        self.cutmix_prob = config_dict['train_cfg'].get('cutmix_prob', 0.5)  # CutMix 通常有一个单独的应用概率
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.normalize = config_dict['train_cfg']['normalize']
        if self.normalize:
            self.mean_data = np.array(config_dict['dataset_cfg']['mean_ampl'])
            self.std_data = np.array(config_dict['dataset_cfg']['std_ampl'])

        self.is_random_chirp = is_random_chirp
        if is_random_chirp:
            self.n_chirps = config_dict['model_cfg']['n_chirps']
            if self.n_chirps > self.dataset.sensor_cfg.n_chirps:
                raise ValueError("n_chirps in model config larger than n_chirps in sensor config")

        # get sequence list
        self.split = split
        self.seq_ids = dataset.seq_sets[split]
        if subset is not None:
            self.seq_ids = self.seq_ids[:subset]

        # get image paths
        self.image_paths = {}
        for seq_id in self.seq_ids:
            self.image_paths[seq_id] = sorted(
                [os.path.join(self.dataset.img_root, seq_id, f) for f in os.listdir(
                    os.path.join(self.dataset.img_root, seq_id)
                ) if f.endswith('.jpg')]
            )

        # get data paths
        self.data_paths = list_pkl_filenames_from_prepared(self.data_dir, self.seq_ids, self.win_size)
        self.n_data = len(self.data_paths)
        print("%s set: %d files" % (split, self.n_data))

    def __len__(self):
        return self.n_data

    def get_data_dict(self, idx):
        """
        Load data dictionary from pkl file
        @param idx: index of the data path
        @return: data dictionary
        """
        data_path = self.data_paths[idx]
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        if data_dict['anno'] is not None:
            # Drop classes that are not in the dataset
            data_dict['anno']['obj_infos'] = self.dataset.drop_invalid_classes_seq(data_dict['anno']['obj_infos'])
            # Generate confidence maps
            data_dict['anno']['confmaps'] = self.dataset.generate_confmaps_seq(data_dict['anno']['obj_infos'])

        # Randomly select chirps
        if self.is_random_chirp:
            chirp_ids = random.sample(range(self.dataset.sensor_cfg.n_chirps), self.n_chirps)
            chirp_ids.sort()
            data_dict['radar_data'] = data_dict['radar_data'][chirp_ids]

        # Convert to tensor
        data_dict['radar_data'] = torch.tensor(data_dict['radar_data']).float()
        if data_dict['anno'] is not None:
            data_dict['anno']['confmaps'] = torch.tensor(data_dict['anno']['confmaps']).float()

        return data_dict

    def __getitem__(self, idx):
        # Step 1: Get data dictionary
        data_dict = self.get_data_dict(idx)
        if data_dict is None:
            return None

        # Step 2: (Original MixUp logic was here)

        # vvvvvvvvvvvv 【修改】 vvvvvvvvvvvv
        # Step 3: Apply CutMix if specified
        if self.use_cutmix and self.split == "train" and random.random() < self.cutmix_prob:
            try:
                # 1. Load a random mixin sample
                mix_idx = random.randint(0, self.n_data - 1)
                mix_data_dict = self.get_data_dict(mix_idx)
                if mix_data_dict is None:
                    raise IOError("Loaded mix_data_dict is None")

                # 2. Generate CutMix bounding box
                # lambda (混合比例) 在 CutMix 中是根据 alpha=1.0 (通常) 抽样
                lambda_ = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

                # radar_data shape is (C, T, H, W)
                # confmaps shape is (n_class, T, H, W)
                # 我们在 H 和 W 维度 (dim 2 和 3) 上进行剪切
                H = data_dict['radar_data'].shape[2]
                W = data_dict['radar_data'].shape[3]

                cut_ratio = np.sqrt(1. - lambda_)
                cut_h = int(H * cut_ratio)
                cut_w = int(W * cut_ratio)

                # 随机选择剪切框的中心点
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                # 计算剪切框的坐标 (并确保不越界)
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                # 3. 将 mix_data_dict 的 patch 粘贴到 data_dict
                #    这同时适用于 radar_data 和 confmaps (标签)
                data_dict['radar_data'][:, :, bby1:bby2, bbx1:bbx2] = mix_data_dict['radar_data'][:, :, bby1:bby2,
                                                                      bbx1:bbx2]

                if data_dict['anno'] is not None and mix_data_dict['anno'] is not None:
                    data_dict['anno']['confmaps'][:, :, bby1:bby2, bbx1:bbx2] = mix_data_dict['anno']['confmaps'][:, :,
                                                                                bby1:bby2, bbx1:bbx2]

                # 注意：CutMix 不需要像 MixUp 一样返回 lambda 来调整损失
                # 它通过直接修改标签热力图来工作

            except (IOError, TypeError, ValueError) as e:
                # 如果加载 mix_data_dict 失败或出现其他问题，则跳过 CutMix
                # print(f"Skipping CutMix due to error: {e}")
                pass
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Step 3.5: Apply other augmentations (mirror, reverse, etc.)
        if self.split == "train" and data_dict['anno'] is not None:
            data_dict['radar_data'], data_dict['anno']['confmaps'], data_dict['image_paths'] = random_apply(
                data_dict['radar_data'],
                data_dict['anno']['confmaps'],
                data_dict['image_paths'],
                self.aug_dict
            )

        # Step 4: Normalize data
        if self.normalize:
            mean_data = self.mean_data.tile(self.n_chirps)
            std_data = self.std_data.tile(self.n_chirps)
            data_dict['radar_data'] = (data_dict['radar_data'] - mean_data[:, None, None, None]) / std_data[:, None,
                                                                                                   None, None]

        # Step 5: Slice the sequence to the last frame if needed (CRITICAL: Do this at the very end)
        if not self.all_confmaps and data_dict['anno'] is not None:
            data_dict['anno']['confmaps'] = data_dict['anno']['confmaps'][:, -1]
            data_dict['anno']['obj_infos'] = data_dict['anno']['obj_infos'][-1]

        # Step 6: Add positional encoding if needed
        if self.model_name == 'UTAE':
            fps = 1 / 30
            pe = np.linspace(0, fps * self.win_size, self.win_size)
            data_dict['pe'] = torch.tensor(pe).float()

        return data_dict