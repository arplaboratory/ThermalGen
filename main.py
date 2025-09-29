import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
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
    
    if hasattr(train_cfg.training, "load") and train_cfg.training.load_type == "finetune":
        model = ThermalGen.load_from_checkpoint(train_cfg.training.load,
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
                                                training_stage=train_cfg.training.training_stage if hasattr(train_cfg.training, "training_stage") else "full",
                                                strict=False)
    else:
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
            training_stage=train_cfg.training.training_stage if hasattr(train_cfg.training, "training_stage") else "full",
            gradient_accumulation=train_cfg.training.gradient_accumulation if hasattr(train_cfg.training, "gradient_accumulation") else 1,
            calculate_stats=train_cfg.training.calculate_stats if hasattr(train_cfg.training, "calculate_stats") else False,
        )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        filename=f'{model.model_arch}' + '_{epoch:02d}_FID[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/FID:.4f}]_LPIPS[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/LPIPS:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    if train_cfg.training.mixed_precision:
        mixed_precision_setting = "16-mixed"
    else:
        mixed_precision_setting = "32"
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=mixed_precision_setting,
        max_epochs=train_cfg.training.num_epochs,
        limit_val_batches=0,
        callbacks=[checkpoint_cb, lr_monitor],# we only run the checkpointing callback (you can add more)
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
    )

    # we call the trainer, we give it the model and the datamodule
    # trainer.validate(model=model, datamodule=datamodule)
    if hasattr(train_cfg.training, "load"):
        print(f"Loading Model: {train_cfg.training.load}")
        if train_cfg.training.load_type == "resume":
            print("RESUME FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=train_cfg.training.load)
        elif train_cfg.training.load_type == "finetune":
            print("FINETUNE FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule)
    else:
        print("Training from scratch")
        trainer.fit(model=model, datamodule=datamodule)

    # torch.distributed.destroy_process_group()
    # if trainer.is_global_zero:
    #     trainer = pl.Trainer(
    #         accelerator='gpu',
    #         devices=1,
    #         default_root_dir=f'./logs/', # Tensorflow can be used to viz 
    #         num_nodes=1,
    #         precision=mixed_precision_setting,
    #         logger=wandb_logger,
    #     )
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path="last")