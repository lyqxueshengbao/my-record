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
    #parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()
deterministic = False


seed = 252 if args.seed is None else args.seed

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

pl.seed_everything(seed=seed, workers=True)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = ROD2021Dataset(config_dict['dataset_cfg'])
train_cfg = config_dict['train_cfg']
model_cfg = config_dict['model_cfg']

model = Model(model_cfg, train_cfg['loss_cfg'], config_dict['dataset_cfg']['object_cfg'],
              train_cfg['lr'], train_cfg['optimizer_cfg'],
              config_dict['dataset_cfg']['win_size'], train_cfg['n_epoch'],
              dataset.max_in_matrix, dataset.max_out_matrix)

logger = TensorBoardLogger(save_dir=os.path.join('logs', config_dict['model_cfg']['name']),
                           name=config_dict['dataset_cfg']['name'])
callbacks = [ModelCheckpoint(dirpath=os.path.join(logger.log_dir, 'checkpoints'),
                             filename='{epoch}-{val_loss:.2f}-{AP:.2f}-{AR:.2f}',
                             monitor='val_loss',
                             mode='min',
                             save_last=True,
                             save_top_k=5),
             EarlyStopping(monitor='val_loss', patience=train_cfg['patience'], mode='min',
                           min_delta=0.001),
             LearningRateMonitor(logging_interval='step')]

if args.resume_ckpt is None:
    if train_cfg['load_pretrain']:
        model.load_pretrain_record(train_cfg['pretrain_path'], train_cfg['pretrain_name'])
else:
    model.load_from_checkpoint(args.resume_ckpt, model_cfg=model_cfg, loss_cfg=train_cfg['loss_cfg'],
                 object_cfg=config_dict['dataset_cfg']['object_cfg'],
                 lr=train_cfg['lr'], optimizer_cfg=train_cfg['optimizer_cfg'],
                 win_size=config_dict['dataset_cfg']['win_size'], n_epoch=train_cfg['n_epoch'],
                 max_in_matrix=dataset.max_in_matrix, max_out_matrix=dataset.max_out_matrix,
                 config_dict=config_dict,
                 cruw_dataset_obj=dataset, save_dir=logger.log_dir)

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'
trainer = pl.Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, strategy='ddp', devices=6,
                     max_epochs=train_cfg['n_epoch'], deterministic=deterministic,
                     gradient_clip_val=1.0) # <--- 新增梯度裁剪

print('Start training')
trainer.fit(model, ckpt_path=args.resume_ckpt)

print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

if args.test_on_val:
    print('Set for evaluation: VALIDATION')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dit, all_confmaps=True)
elif args.test_all:
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path='best')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')
else:
    print('Set for evaluation: TEST')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')