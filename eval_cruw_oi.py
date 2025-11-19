import argparse
import os.path

import yaml
from cruw import CRUW
import torch
import torch.nn as nn  # 新增引用
from torch.utils.data import DataLoader  # 新增引用
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_configs, update_config_dict
from evaluation.eval_cruw import eval_on_test, eval_on_val
from utils.models_utils import get_models
from datasets import ROD2021Dataset
from datasets.cruw.collate_functions import cr_collate  # 新增引用
from executors import RECORDOIExecutor as Model


def parse_args():
    parser = argparse.ArgumentParser(description='RECORD - Evaluate model')
    parser.add_argument('--config', required=True, type=str, help='configuration file path')
    parser.add_argument('--log_dir', required=True, type=str, help='Log directory (e.g. ./logs/)')
    parser.add_argument('--version', required=True, type=str, help='Version of the run to evaluate')
    parser.add_argument('--ckpt', required=True, type=str, help='Ckpt to resume the training')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')

    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


# === 新增：BN 校准函数 ===
def calibrate_bn(model, train_loader, device, num_batches=200):
    """
    使用训练集数据重新校准 BN 层的 running_mean 和 running_var。
    解决 shuffle=False 训练导致的 BN 统计量偏移问题。
    """
    print(f"Running BN calibration for {num_batches} batches...")

    # 1. 确保模型处于 eval 模式 (除了 BN)
    model.eval()

    # 2. 强制所有 BN 层进入训练模式 (为了更新 running stats)
    # 同时冻结参数 (requires_grad=False) 防止更新权重
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.requires_grad_(False)
            # 可选：重置统计量让其从头计算，避免受到错误历史影响
            # m.reset_running_stats()

    # 3. 跑数据
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            # 适配输入维度
            # 如果 Loader 返回的是 Buffer 格式 (B, C, T, H, W)，需要合并 B*T
            if batch['radar_data'].dim() == 5:
                B, C, T, H, W = batch['radar_data'].shape
                # 变成 (B*T, C, H, W) 符合 Online 模型单帧输入的 Stem 要求
                inputs = batch['radar_data'].view(B * T, C, H, W).to(device)
            else:
                # 如果已经是 (B, C, H, W)
                inputs = batch['radar_data'].to(device)

            # 只需要跑过 Stem 部分 (包含 BN 的部分) 即可更新统计量
            # RecordOI 的结构是 model.encoder.forward_stem
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'forward_stem'):
                model.encoder.forward_stem(inputs)
            else:
                # 兼容写法：如果无法定位 stem，就跑一次完整 forward
                # 注意：如果 forward 内部有状态更新逻辑，这里可能会有副作用，
                # 但对于 BN 校准，通常只关心 Stem。
                # 对于 RecordOI，直接 forward 可能会报错 (因为没有 hidden state)，
                # 所以尽量用 forward_stem
                try:
                    model.encoder.forward_stem(inputs)
                except:
                    print("Warning: Could not call forward_stem, skipping batch.")

    print("BN calibration finished.")
    model.eval()  # 恢复全部 eval 模式


# ========================


args = parse_args()

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']
test_cfg = config_dict['test_cfg']
dataset_cfg = config_dict['dataset_cfg']

# Load model
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

# Init CRUW dataset utils
dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
               sensor_config_name=config_dict['model_cfg']['sensor_config'])
radar_configs = dataset.sensor_cfg.radar_cfg
range_grid = dataset.range_grid
angle_grid = dataset.angle_grid
data_dir = config_dict['dataset_cfg']['data_dir']

if args.test_on_val or args.test_all:
    valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict,
                                   all_confmaps=True, split='valid')
else:
    valid_dataset = None

log_dir = args.log_dir
name = model_name
version = args.version
ckpt_path = args.ckpt
logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version)  # default_hp_metric=False

# Update variables with new config dict
if 'RECORD' in model_name:
    backbone_cfg = yaml.load(open(model_cfg['backbone_pth']), yaml.FullLoader)
    config_dict['model_cfg']['layout'] = backbone_cfg

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']

# 初始化 Executor (会自动加载权重，如果 Executor 内部写了 load_state_dict)
# 注意：通常这里的 executor 加载的是初始化权重，真正的训练权重由 trainer.test(ckpt_path=...) 加载
# 但我们需要先加载权重再校准 BN，所以这里可能需要手动加载一次权重用于校准
executor = Model(model=model_instance, train_dataset=None, val_dataset=valid_dataset, config_dict=config_dict,
                 cruw_dataset_obj=dataset, save_dir=logger.log_dir)

# === 加载权重用于校准 ===
print(f"Loading checkpoint for BN calibration: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
# 处理 state_dict key (去除 'model.' 前缀)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        new_state_dict[k[6:]] = v
    else:
        new_state_dict[k] = v
model_instance.load_state_dict(new_state_dict, strict=True)
# =======================

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
    device = torch.device('cuda')
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'
    device = torch.device('cpu')

trainer = pl.Trainer(logger=logger, accelerator=accelerator, devices=1,
                     max_epochs=train_cfg['n_epoch'])

# === 执行 BN 校准 ===
# 1. 准备训练集 Loader (必须 shuffle=True)
print("Preparing calibration dataset...")
# 注意：这里使用 config_dict 创建训练集，确保 split='train'
# is_random_chirp=True 可以增加数据的多样性
calib_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict,
                               split='train', is_random_chirp=True, all_confmaps=False)

calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=True,
                          num_workers=4, collate_fn=cr_collate)

# 2. 将模型移至 GPU
model_instance.to(device)

# 3. 开始校准
calibrate_bn(model_instance, calib_loader, device, num_batches=200)
# ==================

print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

# 注意：因为我们手动修改了 model_instance (更新了 BN 统计量)
# 我们需要确保 trainer.test 使用的是这个“内存中已修改的模型”，而不是重新从 ckpt_path 加载旧模型
# PyTorch Lightning 的 trainer.test(ckpt_path=...) 会重新加载权重，覆盖我们的校准结果！
# 解决方法：不传 ckpt_path 给 trainer.test，而是让它使用当前的 executor (它包含了已校准的 model)

if args.test_on_val:
    print('Set for evaluation: VALIDATION')
    # 这里的 ckpt_path 设为 None，因为我们上面已经手动加载并校准了
    eval_on_val(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path=None)
elif args.test_all:
    # 先测 Validation
    eval_on_val(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path=None)
    # 再测 Test
    eval_on_test(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path=None)
else:
    eval_on_test(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path=None)