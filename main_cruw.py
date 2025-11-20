import argparse
import yaml
from cruw import CRUW
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_configs, update_config_dict, get_models
from datasets import ROD2021Dataset
from evaluation import eval_on_test, eval_on_val
from executors import RECORDExecutor as Model


def parse_args():
    parser = argparse.ArgumentParser(description='RECORD model')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')

    # === 两个不同的加载参数 ===
    parser.add_argument('--resume_ckpt', type=str, help='断点续训：恢复所有状态 (Crash后使用)')
    parser.add_argument('--finetune_from', type=str, help='微调：仅加载权重，重置优化器 (Stage 2 使用)')
    # ========================

    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()
deterministic = False
seed = 252 if args.seed is None else args.seed

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

pl.seed_everything(seed=seed, workers=True)

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']
test_cfg = config_dict['test_cfg']
dataset_cfg = config_dict['dataset_cfg']

# Load model
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

# === 新增：处理微调权重加载 ===
if args.finetune_from:
    print(f">>> [Stage 2] Loading weights from {args.finetune_from}...")
    # 加载权重文件
    checkpoint = torch.load(args.finetune_from, map_location='cpu')

    # 兼容性处理：提取 state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 移除 'model.' 前缀 (如果存在)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:] if k.startswith('model.') else k
        new_state_dict[name] = v

    # 加载到模型 (strict=False 允许一定程度的结构差异，虽然这里应该是一样的)
    model_instance.load_state_dict(new_state_dict, strict=True)
    print(">>> Weights loaded successfully. Starting fresh training (Epoch 0).")
# =============================

# Init CRUW dataset utils
dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
               sensor_config_name=config_dict['model_cfg']['sensor_config'])
data_dir = config_dict['dataset_cfg']['data_dir']

# Load datasets (确保开启全监督 all_confmaps=True)
train_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='train', is_random_chirp=True)
valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='valid')

log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'], version=model_name + '_' + str(seed), name=model_name,
                           default_hp_metric=False)
run_dir = logger.experiment.log_dir
config_dict['train_cfg']['run_dir'] = run_dir

checkpoint_callback = ModelCheckpoint(dirpath=None, monitor='val_loss', mode="min", save_last=True, save_top_k=5)
lr_tracker = LearningRateMonitor()
early_stop = EarlyStopping(monitor='val_loss', patience=7, mode='min')
callbacks = [checkpoint_callback, lr_tracker, early_stop]

if 'RECORD' in model_name:
    backbone_cfg = yaml.load(open(model_cfg['backbone_pth']), yaml.FullLoader)
    config_dict['model_cfg']['layout'] = backbone_cfg

model = Model(model=model_instance, train_dataset=train_dataset, val_dataset=valid_dataset, config_dict=config_dict,
              cruw_dataset_obj=dataset, save_dir=logger.log_dir)

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'

trainer = pl.Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, strategy='ddp', devices=6,
                     max_epochs=train_cfg['n_epoch'], deterministic=deterministic,
                     accumulate_grad_batches=train_cfg['accumulate_grad'])  # 保持 FP16

print('Start training')

# === 修改：仅在断点续训时传入 ckpt_path ===
# 如果是微调，args.resume_ckpt 为空，模型使用上面手动加载的权重从 Epoch 0 开始练
trainer.fit(model, ckpt_path=args.resume_ckpt)
# ========================================

print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

if args.test_on_val:
    print('Set for evaluation: VALIDATION')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True)
elif args.test_all:
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path='best')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')
else:    
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')

print('Training finished.')
