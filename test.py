import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

from thermalgen import ThermalGen
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    wandb_logger = WandbLogger(name=args.config.split('/')[-1].split('.')[0], entity="unistgl", project="ThermalGen")
    datamodule = GenericDataModule(
        datasets_folder=train_cfg.datasets.datasets_folder,
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
        mixed_precision=True if train_cfg.training.mixed_precision else False,
    )
    
    model = ThermalGen(
        #---- Encoder
        model_arch=train_cfg.model.model_arch,
        model_config=train_cfg.model.model_config,
        lr=train_cfg.training.optimizer["lr"],
        optimizer=train_cfg.training.optimizer["name"],
        weight_decay=train_cfg.training.optimizer["weight_decay"], # 0.001 for sgd and 0 for adam,
        momentum=train_cfg.training.optimizer["momentum"],
        lr_sched=train_cfg.training.scheduler["name"],
        lr_sched_args = train_cfg.training.scheduler["args"],

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=train_cfg.training.loss["name"],
        loss_config=train_cfg.training.loss["config"],
        validation_type=train_cfg.training.validation_type
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=f'{train_cfg.datasets.target_val_dataset}_val/FID',
        filename=f'{model.model_arch}' + '_{epoch:02d}_FID[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/FID:.4f}]_LPIPS[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/LPIPS:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        save_last=True,
        mode='min'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    mixed_precision_setting = "32"
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=mixed_precision_setting,
        max_epochs=train_cfg.training.num_epochs,
        check_val_every_n_epoch=train_cfg.training.val_freq, # run validation every epoch
        callbacks=[checkpoint_cb, lr_monitor],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        logger=wandb_logger,
    )

    # we call the trainer, we give it the model and the datamodule
    if not hasattr(train_cfg.training, "load") or train_cfg.training.load == "None":
        print("Warning: Dummy model testing")
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=train_cfg.training.load)