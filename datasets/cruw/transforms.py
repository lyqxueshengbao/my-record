from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
import random


def mirror_horizontal(radar, gt):
    """
    Apply horizontal flipping on radar data and ground truth
    @param radar: radar data
    @param gt: ground truth
    @return: horizontally reversed radar and ground truth data
    """
    radar_tmp, gt_tmp = torch.flip(radar, [2]), torch.flip(gt, [2])
    return radar_tmp, gt_tmp


def mirror_vertical(radar, gt):
    """
    Apply vertical flipping on radar data and ground truth
    @param radar: radar data
    @param gt: ground truth
    @return: vertically reversed radar and ground truth data
    """
    radar_tmp, gt_tmp = torch.flip(radar, [3]), torch.flip(gt, [3])
    return radar_tmp, gt_tmp


def reverse(radar, gt):
    """
    Apply temporal flipping on radar data and ground truth
    @param radar: radar data
    @param gt: ground truth
    @return: temporally reversed radar and ground truth data
    """
    return torch.flip(radar, dims=[1]), torch.flip(gt, dims=[1])


def noise(radar, gt, sigma):
    """
    Add a random gaussian noise with mean=0.0 and std=sigma on input radar frame
    @param radar: radar data
    @param gt: ground truth (unmodified here)
    @param sigma: standard deviation of the gaussian noise
    @return: augmented radar data and ground truth
    """
    noise_val = torch.normal(mean=0.0, std=sigma, size=(1,))
    radar = radar + noise_val
    return radar, gt


def sequence_mixup(radar1, confmap1, radar2, confmap2, alpha=0.2):
    """
    Apply MixUp augmentation on sequence-level radar data and confidence maps
    保持时序完整性的序列级混合

    @param radar1: first radar sequence [C, T, H, W]
    @param confmap1: first confidence map [n_class, T, H, W] or [n_class, H, W]
    @param radar2: second radar sequence [C, T, H, W]
    @param confmap2: second confidence map [n_class, T, H, W] or [n_class, H, W]
    @param alpha: Beta distribution parameter for mixup ratio
    @return: mixed radar data, mixed confidence maps, and lambda value
    """
    # Sample mixing ratio from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Mix radar sequences (element-wise)
    mixed_radar = lam * radar1 + (1 - lam) * radar2

    # Mix confidence maps (element-wise)
    mixed_confmap = lam * confmap1 + (1 - lam) * confmap2

    return mixed_radar, mixed_confmap, lam


def spatial_mixup(radar1, confmap1, radar2, confmap2):
    """
    Apply spatial MixUp on radar data (alternative safer approach)
    在距离维度上进行空间区域混合

    @param radar1: first radar sequence [C, T, H, W]
    @param confmap1: first confidence map [n_class, T, H, W] or [n_class, H, W]
    @param radar2: second radar sequence [C, T, H, W]
    @param confmap2: second confidence map [n_class, T, H, W] or [n_class, H, W]
    @return: spatially mixed radar data and confidence maps
    """
    _, _, h, w = radar1.shape

    # Randomly select cut position in range dimension (width)
    cut_w = np.random.randint(w // 4, 3 * w // 4)

    # Clone to avoid in-place modification
    mixed_radar = radar1.clone()
    mixed_confmap = confmap1.clone()

    # Mix spatial regions
    mixed_radar[..., cut_w:] = radar2[..., cut_w:]
    mixed_confmap[..., cut_w:] = confmap2[..., cut_w:]

    return mixed_radar, mixed_confmap


def random_apply(radar, gt, image_paths, aug_dict=None):
    """
    Randomly apply data augmentation operation on input radar frames and ground truth
    @param radar: radar data
    @param gt: ground truth
    @param image_paths: list of image paths (only for temporal flipping and visualisation purpose)
    @param aug_dict: dictionary with the augmentation to apply
    @return: augmented radar, ground truth and images paths
    """
    if aug_dict is None:
        aug_dict = {'mirror': 0.5,
                    'reverse': 0.5
                    }

    augmentation_func = []
    for key in aug_dict.keys():
        random_val = random.random()
        prob_key = aug_dict[key]
        if prob_key > random_val:
            augmentation_func.append(key)

    for augmentation_technique in augmentation_func:
        if augmentation_technique == 'mirror':
            random_val = random.randint(-1, 1)
            if random_val == 0:
                radar, gt = mirror_horizontal(radar, gt)
            elif random_val == 1:
                radar, gt = mirror_vertical(radar, gt)
            else:
                tmp_radar, tmp_gt = mirror_horizontal(radar, gt)
                radar, gt = mirror_vertical(tmp_radar, tmp_gt)

        if augmentation_technique == 'gaussian':
            sigma = np.random.uniform(0, 0.03)
            radar, gt = noise(radar, gt, sigma)

        if augmentation_technique == 'reverse':
            radar, gt = reverse(radar, gt)
            image_paths.reverse()

    return radar, gt, image_paths
