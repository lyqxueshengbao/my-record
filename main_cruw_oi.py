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
# 注意：这里导入的是你之前修改过支持 TBPTT 的 Executor
from executors import RECORDOIExecutor as Model


def parse_args():
    parser = argparse.ArgumentParser(description='RECORD model - Online Fine-tuning')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')

    # [修改点 1] 添加加载 Buffer 权重的参数
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the Buffer-mode trained .pth file')

    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser.add_argument('--tb_version', type=str, help='Name to the saved model')
    parser = parse_configs(parser)
    parser = parse_transforms(parser)
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

# ---------------------------------------------------------------------------
# [修改点 2] 加载模型并加载 Buffer 权重
# ---------------------------------------------------------------------------
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

if args.pretrained_path is not None:
    print(f"Loading Buffer-mode weights from: {args.pretrained_path}")
    # 加载权重
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')

    # 处理 checkpoint 格式差异 (有的 checkpoint 把权重放在 'state_dict' 键下)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 移除可能的 'model.' 前缀 (如果 Buffer 训练时用了 Executor 封装保存)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    # 加载权重 (strict=True 确保完全匹配，因为我们架构是对齐的)
    model_instance.load_state_dict(new_state_dict, strict=True)
    print("Weights loaded successfully!")

# ---------------------------------------------------------------------------
# [修改点 3] 冻结 Backbone (Stem)
# ---------------------------------------------------------------------------
# 我们只微调 LSTM 和 Decoder，保护 CNN 特征不被破坏
print("Freezing Backbone (Stem) parameters...")
for param in model_instance.encoder.stem.parameters():
    param.requires_grad = False

# ---------------------------------------------------------------------------

# Init CRUW dataset utils
dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
               sensor_config_name=config_dict['model_cfg']['sensor_config'])
radar_configs = dataset.sensor_cfg.radar_cfg
range_grid = dataset.range_grid
angle_grid = dataset.angle_grid
data_dir = config_dict['dataset_cfg']['data_dir']

# Load datasets
train_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='train')
valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='valid')

log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'], name=model_name, version=args.tb_version,
                           default_hp_metric=False)

# Add some entries to the configuration dict to get back the logs
run_dir = logger.experiment.log_dir
config_dict['train_cfg']['run_dir'] = run_dir

checkpoint_callback = ModelCheckpoint(dirpath=None, monitor='val_loss', mode="min", save_last=True, save_top_k=3)
lr_tracker = LearningRateMonitor()

# 微调通常收敛很快，可以适当减少 patience
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
callbacks = [checkpoint_callback, lr_tracker, early_stop]

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']

model = Model(model=model_instance, train_dataset=train_dataset, val_dataset=valid_dataset, config_dict=config_dict,
              cruw_dataset_obj=dataset, save_dir=logger.log_dir)

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'

# ---------------------------------------------------------------------------
# [修改点 4] 设置 TBPTT (Truncated Backpropagation Through Time)
# ---------------------------------------------------------------------------
# truncated_bptt_steps: 这个值最好等于你 Buffer 训练时的 win_size (例如 16)
# 这意味着每 16 步截断一次梯度，但隐状态会一直传递下去
# max_epochs: 微调不需要跑太多轮，3-5 轮通常足够
print(f"Starting Fine-tuning with TBPTT steps = {train_cfg['win_size']}")

trainer = pl.Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, devices=1,
                     max_epochs=6,  # 建议写死一个小数字，或者在 config 里改小
                     deterministic=deterministic,
                     accumulate_grad_batches=train_cfg['accumulate_grad'],
                     gradient_clip_val=1.0,
                     # 关键参数！开启 TBPTT
                     # truncated_bptt_steps=train_cfg['win_size']
                     )

print('Start training (Fine-tuning)')
# 注意：这里不要用 ckpt_path=args.resume_ckpt，除非你想恢复中断的微调
# 我们已经手动加载了 pretrained_path
trainer.fit(model)

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