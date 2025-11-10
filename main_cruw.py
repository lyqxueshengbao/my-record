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
    # 🔧 修改1: 恢复 --deterministic 参数
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()

# 🔧 修改2: 使用命令行参数控制 deterministic，默认为 True
deterministic = args.deterministic if hasattr(args, 'deterministic') else True

# 🔧 修改3: 如果启用deterministic，设置更严格的种子控制
seed = 252 if args.seed is None else args.seed

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

# 🔧 修改4: 更严格的随机种子设置
pl.seed_everything(seed=seed, workers=True)

if deterministic:
    # 额外的确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"⚙️  已启用确定性模式 (seed={seed})")
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"⚡ 已启用高性能模式 (seed={seed})")

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

# Load datasets
train_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=False,
                               split='train')
valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=False,
                               split='valid')

log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'], version=model_name+'_'+str(seed), name=model_name, default_hp_metric=False)

# Add some entries to the configuration dict to get back the logs
run_dir = logger.experiment.log_dir
config_dict['train_cfg']['run_dir'] = run_dir

checkpoint_callback = ModelCheckpoint(dirpath=None, monitor='val_loss', mode="min", save_last=True, save_top_k=5)
lr_tracker = LearningRateMonitor()

# 🔧 修改5: 增加早停的耐心值，避免过早停止
early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=True)
callbacks = [checkpoint_callback, lr_tracker, early_stop]

# Update variables with new config dict
if 'RECORD' in model_name:
    backbone_cfg = yaml.load(open(model_cfg['backbone_pth']), yaml.FullLoader)
    config_dict['model_cfg']['layout'] = backbone_cfg

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

# 🔧 修改6: 添加梯度裁剪和异常检测
trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    accelerator=accelerator,
    strategy='ddp',
    devices=6,
    max_epochs=train_cfg['n_epoch'],
    deterministic=deterministic,
    gradient_clip_val=1.0,  # 梯度裁剪，防止梯度爆炸
    gradient_clip_algorithm='norm',
    detect_anomaly=False,  # 生产环境关闭，调试时可开启
    # 🔧 修改7: 添加精度控制
    precision=32  # 使用完整精度，避免混合精度导致的不稳定
)

print(f'🚀 开始训练 (Seed: {seed}, Deterministic: {deterministic})')
print(f'📊 训练配置:')
print(f'   - Epochs: {train_cfg["n_epoch"]}')
print(f'   - 梯度裁剪: 1.0')
print(f'   - 早停耐心值: 15')
print(f'   - 精度: 32-bit')

try:
    trainer.fit(model, ckpt_path=args.resume_ckpt)
    print("✅ 训练成功完成")
except Exception as e:
    print(f"❌ 训练过程出现异常: {e}")
    import traceback
    traceback.print_exc()
    raise

print("🔍 开始评估")
data_root = config_dict['dataset_cfg']['data_root']

if args.test_on_val:
    print('📌 评估集: VALIDATION')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True)
elif args.test_all:
    print('📌 评估集: VALIDATION + TEST')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path='best')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')
else:
    print('📌 评估集: TEST')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')

print('🎉 训练完成!')