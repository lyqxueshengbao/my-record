import argparse
import yaml
from cruw import CRUW
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_configs, parse_transforms, update_config_dict, get_models
from datasets import ROD2021Dataset
from evaluation import eval_on_test, eval_on_val
from executors import RECORDOIExecutor as Model


def parse_args():
    parser = argparse.ArgumentParser(description='RECORD model')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser.add_argument('--tb_version', type=str, help='Name to the saved model')
    parser = parse_configs(parser)
    parser = parse_transforms(parser)
    args = parser.parse_args()
    return args


args = parse_args()

# Load configs
config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

if args.seed is not None:
    pl.seed_everything(args.seed, workers=True)
    deterministic = True
else:
    deterministic = False

# 1. 实例化 Online 模型
model_cfg = config_dict['model_cfg']
model_instance = get_models(model_cfg)

# =================================================================================
# [新增修改] 加载 Buffer 模式预训练权重 (Two-step Strategy)
# =================================================================================
# 请将下面的路径修改为你 Buffer 模式训练出的最佳权重文件路径 (.ckpt)
buffer_ckpt_path = '/home/liyuqin/projects/record/logs/rod2021_focaloss+mixednorm/RECORD/RECORD_1/checkpoints/last.ckpt'  # <--- 修改这里！

if os.path.exists(buffer_ckpt_path):
    print(f"\n[Info] Loading pretrained weights from Buffer model: {buffer_ckpt_path}")
    # 加载 checkpoint
    checkpoint = torch.load(buffer_ckpt_path, map_location='cpu')

    # 提取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 处理键名不匹配问题 (Buffer 模型的 key 通常带有 'model.' 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            # 去掉 'model.' 前缀，因为这里的 model_instance 是纯 PyTorch Module
            new_key = k[6:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 加载权重到 Online 模型
    # strict=False 是为了防止有些不重要的 key (比如 loss 相关的) 不匹配导致报错
    missing_keys, unexpected_keys = model_instance.load_state_dict(new_state_dict, strict=False)

    print(f"[Info] Weights loaded successfully!")
    if len(missing_keys) > 0:
        print(f"       Missing keys (usually fine for online specific layers): {len(missing_keys)}")
    if len(unexpected_keys) > 0:
        print(f"       Unexpected keys: {len(unexpected_keys)}")
    print("=================================================================================\n")

else:
    print(f"\n[Warning] Buffer checkpoint NOT found at {buffer_ckpt_path}. Training from scratch...\n")
# =================================================================================


# Create dataloaders
# 1. 实例化 CRUW 对象 (ROD2021Dataset 需要这个对象来读取传感器配置)
cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                    sensor_config_name='sensor_config_rod2021')

# 2. 获取 prepared data 的路径
data_dir = config_dict['dataset_cfg']['data_root']

# 3. 正确实例化 Dataset (补上 data_dir 和 dataset 参数)
train_dataset = ROD2021Dataset(data_dir=data_dir,
                               dataset=cruw_dataset,
                               config_dict=config_dict,
                               split='train')

valid_dataset = ROD2021Dataset(data_dir=data_dir,
                               dataset=cruw_dataset,
                               config_dict=config_dict,
                               split='valid')

# Create executor
executor = Model(model=model_instance, train_dataset=train_dataset, val_dataset=valid_dataset,
                 config_dict=config_dict,
                 cruw_dataset_obj=train_dataset.cruw_dataset)
logger = TensorBoardLogger('lightning_logs', name=config_dict['name_experiment'], version=args.tb_version)

# Create Checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=logger.log_dir + '/checkpoints',
    filename='record-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

# Create Early Stopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=config_dict['train_cfg']['patience'],
    verbose=False,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]
train_cfg = config_dict['train_cfg']
if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    accelerator=accelerator,
    strategy='ddp',
    devices=6,  # 根据你的 GPU 数量调整
    max_epochs=train_cfg['n_epoch'],
    deterministic=deterministic,
    accumulate_grad_batches=train_cfg['accumulate_grad']
)

print('Start training')
trainer.fit(model=executor, ckpt_path=args.resume_ckpt)

print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

# if args.test_on_val:
#     print('Set for evaluation: VALIDATION')
#     eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
#                 config_dict=config_dict, all_confmaps=True)
# elif args.test_all:
#     eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
#                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')
#     eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
#                 config_dict=config_dict, all_confmaps=True)