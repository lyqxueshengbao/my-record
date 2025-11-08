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
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrained model')
    parser.add_argument('--pretrain_path', type=str, help='Path to pretrained model')
    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()

# 设置随机种子
seed = 252 if args.seed is None else args.seed

# 加载配置文件
config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

# 设置种子和确定性
pl.seed_everything(seed=seed, workers=True)

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Deterministic mode enabled')

# 提取配置
model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']
test_cfg = config_dict['test_cfg']
dataset_cfg = config_dict['dataset_cfg']

# 初始化 CRUW 数据集工具（采用第一段的方式，更灵活）
dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
               sensor_config_name=config_dict['model_cfg']['sensor_config'])
radar_configs = dataset.sensor_cfg.radar_cfg
range_grid = dataset.range_grid
angle_grid = dataset.angle_grid
data_dir = config_dict['dataset_cfg']['data_dir']

# 加载数据集
train_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict,
                               all_confmaps=False, split='train')
valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict,
                               all_confmaps=False, split='valid')

# 加载模型
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

# 设置日志目录
log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'],
                          version=model_name+'_'+str(seed),
                          name=model_name,
                          default_hp_metric=False)

# 添加日志目录到配置字典
run_dir = logger.experiment.log_dir
config_dict['train_cfg']['run_dir'] = run_dir

# 配置回调函数（更详细的checkpoint配置）
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logger.log_dir, 'checkpoints'),
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_last=True,
    save_top_k=5
)
lr_tracker = LearningRateMonitor(logging_interval='step')
early_stop = EarlyStopping(monitor='val_loss', patience=train_cfg.get('patience', 7),
                          mode='min', min_delta=0.001)
callbacks = [checkpoint_callback, lr_tracker, early_stop]

# 更新backbone配置（如果是RECORD模型）
if 'RECORD' in model_name:
    backbone_cfg = yaml.load(open(model_cfg['backbone_pth']), yaml.FullLoader)
    config_dict['model_cfg']['layout'] = backbone_cfg

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']

# 初始化模型
model = Model(model=model_instance,
             train_dataset=train_dataset,
             val_dataset=valid_dataset,
             config_dict=config_dict,
             cruw_dataset_obj=dataset,
             save_dir=logger.log_dir)

# 加载预训练模型（如果需要）
if args.resume_ckpt is None:
    if args.load_pretrain or train_cfg.get('load_pretrain', False):
        pretrain_path = args.pretrain_path or train_cfg.get('pretrain_path')
        pretrain_name = train_cfg.get('pretrain_name', 'pretrained_model')
        if pretrain_path and os.path.exists(pretrain_path):
            print(f'Loading pretrained model from: {pretrain_path}')
            model.load_pretrain_record(pretrain_path, pretrain_name)
        else:
            print('Warning: Pretrain path not found, training from scratch')
else:
    print(f'Resuming training from checkpoint: {args.resume_ckpt}')

# 设置训练设备
if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'

# 初始化训练器（添加梯度裁剪）
trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    accelerator=accelerator,
    strategy=DDPStrategy(
        gradient_as_bucket_view=False,  # 解决梯度 strides 不匹配警告
        find_unused_parameters=False     # 如果有未使用的参数设为 True
    ),
    devices=6,
    max_epochs=train_cfg['n_epoch'],
    deterministic=deterministic,
    gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0)  # 添加梯度裁剪
)

# 开始训练
print('Start training')
print(f'Model: {model_name}, Seed: {seed}')
trainer.fit(model, ckpt_path=args.resume_ckpt)

# 开始评估
print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

if args.test_on_val:
    print('Set for evaluation: VALIDATION')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset,
               data_root=data_root, config_dict=config_dict,
               all_confmaps=True, ckpt_path='best')
elif args.test_all:
    print('Set for evaluation: VALIDATION + TEST')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset,
               data_root=data_root, config_dict=config_dict,
               all_confmaps=True, ckpt_path='best')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset,
                data_root=data_root, config_dict=config_dict,
                all_confmaps=True, ckpt_path='best')
else:
    print('Set for evaluation: TEST')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset,
                data_root=data_root, config_dict=config_dict,
                all_confmaps=True, ckpt_path='best')

print('Training finished.')
