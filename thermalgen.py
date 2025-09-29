import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import torchvision
import torch.nn.functional as F
import os
import numpy as np
import itertools

from utils.losses import get_loss
from utils.metrics import calculate_psnr, calculate_ssim, calculate_fid, calculate_lpips
from models.generative_models.pix2pix_networks.networks import UnetGenerator, NLayerDiscriminator, get_norm_layer, ResnetGenerator
from models.generative_models.pix2pixHD_networks.networks import define_G, define_D
from models.generative_models.vqgan_networks.networks import VQGAN
from diffusers.models import AutoencoderKL, AutoencoderDC
from models.generative_models.sit_networks import sit_networks
from models.generative_models.sit_networks.transport import create_transport, Sampler
from dataloaders.GenericDataloader import IMAGENET_MEAN_STD, NORMAL_MEAN_STD
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from collections import OrderedDict
import copy

class ThermalGen(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        model_arch='pix2pix',
        model_config={
            "G_arch": "unet",
            "D_arch": "patchGAN",
        },
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
        },
        
        #----- Loss
        loss_name='pix2pix', 
        loss_config = {
            'G_mode': 'lsgan',
            'G_loss_lambda': 100.0,
        },
        training_stage='full',
        gradient_accumulation=1,
        calculate_stats=False,
        validation_type="ema",
    ):
        super().__init__()

        # Disable Auto Optim for GAN
        self.automatic_optimization = False
        # Backbone
        self.model_arch = model_arch
        self.model_config = model_config

        # Train hyperparameters
        self.lr = float(lr)
        self.optimizer = optimizer
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.loss_config = loss_config
        self.training_stage = training_stage
        self.gradient_accumulation = gradient_accumulation
        self.calculate_stats = calculate_stats
        self.validation_type = validation_type
        
        self.save_hyperparameters() # write hyperparams into a file
        
        if self.loss_name == "pix2pix" or self.loss_name == "cyclegan" or self.loss_name == "pix2pixHD":
            self.loss_fn_GAN, self.loss_fn_L1 = get_loss(loss_name, loss_config)
        else:
            self.loss_fn = get_loss(loss_name, loss_config)
        
        # ----------------------------------
        # get the backbone and the aggregator
        if self.model_arch == "pix2pix":
            norm_layer = get_norm_layer(norm_type=self.model_config['GAN_norm'])
            if self.model_config["G_arch"] == "unet":
                self.model = UnetGenerator(3, 1, 8, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'])
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator = NLayerDiscriminator(3+1, norm_layer=norm_layer)
            else:
                raise NotImplementedError()
        elif self.model_arch == "cyclegan":
            norm_layer = get_norm_layer(norm_type=self.model_config['GAN_norm'])
            if self.model_config["G_arch"] == "resnet":
                self.model_A = ResnetGenerator(1, 3, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'], n_blocks=9)
                self.model = ResnetGenerator(3, 1, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'], n_blocks=9)
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator_A = NLayerDiscriminator(3, norm_layer=norm_layer)
                self.discriminator_B = NLayerDiscriminator(1, norm_layer=norm_layer)
            else:
                raise NotImplementedError()
        elif self.model_arch == "pix2pixHD":
            if self.model_config["G_arch"] == "global":
                self.model = define_G(3, 1, 64, "global", n_downsample_global=self.model_config["n_downsample_global"], norm=self.model_config['GAN_norm'], upsample=self.model_config['GAN_upsample'])
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator = define_D(3+1, 64, self.model_config["n_layers_D"], num_D=self.model_config["num_D"], getIntermFeat=True, norm=self.model_config['GAN_norm'])
            else:
                raise NotImplementedError()
        elif self.model_arch == "vqgan":
            self.model = VQGAN(self.model_config)
            self.model.divisible = model_config['divisible']
            self.output_type = "normal"
        elif self.model_arch == "klvae" or self.model_arch == "klvae_RGB":
            divisible = self.model_config['divisible']
            self.model_config.pop('divisible')
            self.model = AutoencoderKL(**self.model_config)
            self.model.divisible = divisible
            self.output_type = "normal"
        elif self.model_arch == "dcae":
            divisible = self.model_config['divisible']
            self.model_config.pop('divisible')
            self.model = AutoencoderDC(**self.model_config)
            self.model.divisible = divisible
            self.output_type = "normal"
        elif self.model_arch == "sit":
            num_classes = 1000
            if self.model_config['vae_model'] == "klvae":
                self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=4, num_classes=num_classes, injection_args=self.model_config['injection_args'], learn_sigma=True if 'pretrain_load' in self.model_config else False, repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                self.thermal_vae = AutoencoderKL(**self.model_config['vae_config'])
                self.thermal_vae = self.load_pretrained(self.thermal_vae)
                self.RGB_vae = AutoencoderKL().from_pretrained(f"stabilityai/sd-vae-ft-{self.model_config['vae']}")
            elif self.model_config['vae_model'] == "dcae":
                self.thermal_vae = AutoencoderDC(**self.model_config['vae_config'])
                self.thermal_vae = self.load_pretrained(self.thermal_vae)
                if model_config['divisible'] == 32:
                    self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=32, num_classes=num_classes, injection_args=self.model_config['injection_args'], repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                    self.RGB_vae = AutoencoderDC().from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
                elif model_config['divisible'] == 64:
                    self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=128, num_classes=num_classes, injection_args=self.model_config['injection_args'], repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                    self.RGB_vae = AutoencoderDC().from_pretrained(f"mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers")
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            if self.model.repa:
                self.repa_encoder = self.load_encoder(enc_type="dinov2")
                self.repa_input = model_config['repa_input']
            self.transport = create_transport(**self.model_config['transport_config'])
            self.sampler = Sampler(self.transport)
            self.ema = copy.deepcopy(self.model)
            self.use_cfg = self.model_config['cfg_scale'] > 1.0
            self.thermal_normalizer = self.model_config['thermal_normalizer'] if 'thermal_normalizer' in self.model_config else None
            self.RGB_normalizer = self.model_config['RGB_normalizer'] if 'RGB_normalizer' in self.model_config else None
            self.model.divisible = model_config['divisible']
            self.RGB_encoder_training = model_config['RGB_encoder_training'] if 'RGB_encoder_training' in model_config else False
            self.style_finetuning = model_config['style_finetuning'] if 'style_finetuning' in model_config else False
            self.kl_training = model_config['kl_training'] if 'kl_training' in model_config else False
            if self.kl_training:
                assert self.RGB_encoder_training and self.model_config['vae_model'] == "klvae" # kl training should only be used when RGB encoder is training
            if 'cache_rate' in self.model_config and self.model_config['cache_rate'] > 1:
                self.latent_cache = []
                if not self.RGB_encoder_training:
                    self.RGB_latent_cache = []
                else:
                    self.RGB_cache = []
                self.latent_cache_init = False
            self.output_type = "normal"
        else:
            self.output_type = "normal"
            raise NotImplementedError()

        # For validation in Lightning v2.0.0
        self.log_img_first_iter_train = False
        self.log_img_first_iter_val = False
        self.log_img_first_iter_test = False

    # the forward pass of the lightning model
    def forward(self, RGB, dataset_idx=None, Thermal=None, Training=False):
        # Pad RGB and Thermal
        RGB_padded = self.pad_to_divisble(RGB, multiple=self.model.divisible)
        if Thermal is not None:
            Thermal_padded = self.pad_to_divisble(Thermal, multiple=self.model.divisible)
        else:
            Thermal_padded = None

        if self.model_arch == "pix2pix" or self.model_arch == "pix2pixHD":
            Pred_Thermal_padded = self.model(RGB_padded)
            results = Pred_Thermal_padded
        elif self.model_arch == "cyclegan":
            if Training:
                assert Thermal is not None
                Thermal_pred_padded = self.model(RGB_padded)
                RGB_rec_padded = self.model_A(Thermal_pred_padded)
                RGB_pred_padded = self.model_A(Thermal_padded)
                Thermal_rec_padded = self.model(RGB_pred_padded)
                results = [Thermal_pred_padded, RGB_rec_padded, Thermal_rec_padded, RGB_pred_padded]
            else: # Evaluation
                Pred_Thermal_padded = self.model(RGB_padded)
                results = Pred_Thermal_padded
        elif self.model_arch == "vqgan":
            if Training:
                Pred_Thermal_padded, qloss = self.model(RGB_padded)
                results = [Pred_Thermal_padded, qloss]
            else:
                Pred_Thermal_padded, _ = self.model(RGB_padded)
                results = Pred_Thermal_padded
        elif self.model_arch == "klvae":
            if Training:
                posterior = self.model.encode(Thermal_padded).latent_dist
                z = posterior.sample(generator=None)
                qloss = posterior.kl()
                Pred_Thermal_padded = self.model.decode(z).sample
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = [Pred_Thermal_padded, qloss]
            else:
                Pred_Thermal_padded = self.model(Thermal_padded).sample
                results = Pred_Thermal_padded
        elif self.model_arch == "klvae_RGB":
            if Training:
                posterior = self.model.encode(RGB_padded).latent_dist
                z = posterior.sample(generator=None)
                qloss = posterior.kl()
                Pred_RGB_padded = self.model.decode(z).sample
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = [Pred_RGB_padded, qloss]
            else:
                Pred_RGB_padded = self.model(RGB_padded).sample
                results = Pred_RGB_padded
        elif self.model_arch == "dcae":
            if Training:
                z = self.model.encode(Thermal_padded, return_dict=False)[0]
                Pred_Thermal_padded = self.model.decode(z, return_dict=False)[0]
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = Pred_Thermal_padded
            else:
                Pred_Thermal_padded = self.model(Thermal_padded).sample
                results = Pred_Thermal_padded
        elif self.model_arch == "sit":
            if Training:
                if self.model.repa:
                    with torch.no_grad():
                        zs = []
                        if self.repa_input == "RGB":
                            RGB_norm = self.preprocess_raw_image(RGB_padded, enc_type="dinov2")
                            z = self.repa_encoder.forward_features(RGB_norm)
                        elif self.repa_input == "Thermal":
                            Thermal_norm = self.preprocess_raw_image(Thermal_padded, enc_type="dinov2")
                            z = self.repa_encoder.forward_features(Thermal_norm)
                        z = z['x_norm_patchtokens']
                        zs.append(z)
                else:
                    zs = None
                if 'vae_mixed_precision' in self.model_config and self.model_config['vae_mixed_precision']:
                    with torch.autocast("cuda", torch.float16):
                        input_latent = self.generate_latent_train(Thermal_padded, RGB_padded)
                else:
                    input_latent = self.generate_latent_train(Thermal_padded, RGB_padded)
                x, x_RGB = input_latent[0], input_latent[1]
                if self.calculate_stats:
                    self.latent_list = torch.cat([self.latent_list, x.detach().cpu()])
                    self.latent_RGB_list = torch.cat([self.latent_RGB_list, x_RGB.detach().cpu()])
                model_kwargs = dict(y=dataset_idx, x_RGB=x_RGB)
                loss_dict = self.transport.training_losses(self.model, x, model_kwargs, zs)
                results = loss_dict["loss"]
                if self.kl_training:
                    results += self.model_config['kl_weight'] * input_latent[2] # kl loss
            else:
                if self.model_config['vae_model'] == "klvae":
                    latent_size = RGB_padded.shape[2]//8, RGB_padded.shape[3]//8
                    zs = torch.randn(RGB.shape[0], 4, latent_size[0], latent_size[1], device="cuda")
                elif self.model_config['vae_model'] == "dcae":
                    if self.model_config['divisible'] == 32:
                        latent_size = RGB_padded.shape[2]//32, RGB_padded.shape[3]//32
                        zs = torch.randn(RGB.shape[0], 32, latent_size[0], latent_size[1], device="cuda")
                    elif self.model_config['divisible'] == 64:
                        latent_size = RGB_padded.shape[2]//64, RGB_padded.shape[3]//64
                        zs = torch.randn(RGB.shape[0], 128, latent_size[0], latent_size[1], device="cuda")
                else:
                    raise NotImplementedError()
                ys = dataset_idx
                sample_fn = self.sampler.sample_ode()
                with torch.no_grad():
                    if self.model_config['vae_model'] == "klvae":
                        x_RGB = self.RGB_vae.encode(RGB_padded).latent_dist.sample()
                    elif self.model_config['vae_model'] == "dcae":
                        x_RGB = self.RGB_vae.encode(RGB_padded).latent
                    if self.RGB_normalizer is not None:
                        x_RGB = x_RGB.mul_(self.RGB_normalizer)
                    if self.use_cfg:
                        zs = torch.cat([zs, zs], 0)
                        y_null = torch.tensor([1000] * len(ys), device="cuda")
                        ys = torch.cat([ys, y_null], 0)
                        x_RGB = torch.cat([x_RGB, x_RGB], 0)
                        sample_model_kwargs = dict(y=ys, x_RGB=x_RGB, cfg_scale=self.model_config['cfg_scale'])
                        if self.validation_type == "ema":
                            model_eval = self.ema.forward_with_cfg
                        elif self.validation_type == 'current':
                            model_eval = self.model.forward_with_cfg
                        else:
                            raise NotImplementedError()
                    else:
                        if 'force_un' in self.model_config and self.model_config['force_un']:
                            y_null = torch.tensor([1000] * len(ys), device="cuda")
                            sample_model_kwargs = dict(y=y_null, x_RGB=x_RGB)
                        else:
                            sample_model_kwargs = dict(y=ys, x_RGB=x_RGB)
                        if self.validation_type == "ema":
                            model_eval = self.ema.forward
                        elif self.validation_type == 'current':
                            model_eval = self.model.forward
                        else:
                            raise NotImplementedError()
                    samples = sample_fn(zs, model_eval, **sample_model_kwargs)[-1]
                    if self.use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    Pred_Thermal_padded = self.thermal_vae.decode(samples / self.thermal_normalizer).sample
                    results = Pred_Thermal_padded
        else:
            raise NotImplementedError()

        # Reverse Pad RGB and Thermal
        if type(results) != list and len(results.shape) == 4:
            results = results[:, :, :RGB.shape[2], :RGB.shape[3]]
        else:
            for i, result_padded in enumerate(results):
                if len(result_padded.shape) == 4:
                    results[i] = result_padded[:, :, :RGB.shape[2], :RGB.shape[3]]
        return results
    
    def generate_latent_train(self, Thermal_padded, RGB_padded):
        with torch.no_grad():
            if self.model_config['vae_model'] == "klvae":
                if self.thermal_vae.in_channels == 3:
                    Thermal_padded = Thermal_padded.repeat(1,3,1,1)
                x = self.thermal_vae.encode(Thermal_padded).latent_dist.sample()
            elif self.model_config['vae_model'] == "dcae":
                x = self.thermal_vae.encode(Thermal_padded).latent
            if self.thermal_normalizer is not None:
                x = x.mul_(self.thermal_normalizer)
            if hasattr(self, 'latent_cache'):
                for item in x.detach().cpu():
                    self.latent_cache.append(item)
        if self.RGB_encoder_training:
            if self.model_config['vae_model'] == "klvae":
                posterior = self.RGB_vae.encode(RGB_padded).latent_dist
                x_RGB = posterior.sample()
            elif self.model_config['vae_model'] == "dcae":
                x_RGB = self.RGB_vae.encode(RGB_padded).latent
            if self.RGB_normalizer is not None:
                x_RGB = x_RGB.mul_(self.RGB_normalizer)
            if hasattr(self, 'RGB_cache'):
                for item in RGB_padded.detach().cpu():
                    self.RGB_cache.append(item)
        else:
            with torch.no_grad():
                if self.model_config['vae_model'] == "klvae":
                    x_RGB = self.RGB_vae.encode(RGB_padded).latent_dist.sample()
                elif self.model_config['vae_model'] == "dcae":
                    x_RGB = self.RGB_vae.encode(RGB_padded).latent
                if self.RGB_normalizer is not None:
                    x_RGB = x_RGB.mul_(self.RGB_normalizer)
            if hasattr(self, 'RGB_latent_cache'):
                for item in x_RGB.detach().cpu():
                    self.RGB_latent_cache.append(item)
        return x, x_RGB, posterior.kl() if self.kl_training else None
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            opt = torch.optim.SGD
        elif self.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW
        elif self.optimizer.lower() == "adam":
            opt = torch.optim.Adam
        else:
            raise NotImplementedError("Optimizer name is unavailable")
        
        optimizers = []
        if self.model_arch == "pix2pix" or self.model_arch == "pix2pixHD":
            optimizers.append(opt(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
            optimizers.append(opt(
                self.discriminator.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "cyclegan":
            optimizers.append(opt(
                itertools.chain(self.model_A.parameters(), self.model.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
            optimizers.append(opt(
                itertools.chain(self.discriminator_A.parameters(), self.discriminator_B.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "vqgan" or self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae":
            if self.training_stage == "full":
                optimizers.append(opt(
                    self.model.parameters(),
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
                if self.model_arch == "vqgan":
                    optimizers.append(opt(
                        self.loss_fn.discriminator.parameters(),
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
            elif self.training_stage == "mid" and self.model_arch == "dcae":
                optimizers.append(opt(
                    [{'params': self.model.encoder.conv_out.parameters()}, {'params': self.model.decoder.conv_in.parameters()}],
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
            elif self.training_stage == "last" and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae"):
                if self.model_arch == "klvae" or self.model_arch == "klvae_RGB":
                    optimizers.append(opt(
                        [{'params': self.model.decoder.up_blocks[3].parameters()}, {'params': self.model.decoder.conv_norm_out.parameters()}, {'params': self.model.decoder.conv_out.parameters()}],
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
                elif self.model_arch == "dcae":
                    optimizers.append(opt(
                        [{'params': self.model.decoder.up_blocks[0].parameters()}, {'params': self.model.decoder.norm_out.parameters()}, {'params': self.model.decoder.conv_out.parameters()}],
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
                optimizers.append(opt(
                    self.loss_fn.discriminator.parameters(),
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
            else:
                raise NotImplementedError()
        elif self.model_arch == "sit":
            if self.style_finetuning:
                optimizers.append(opt(
                    self.model.y_embedder.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                ))
            elif self.RGB_encoder_training:
                optimizers.append(opt(
                    [{'params': self.model.parameters()}, {'params': self.RGB_vae.parameters()}],
                    lr=self.lr,
                    weight_decay=self.weight_decay
                ))
            else:
                optimizers.append(opt(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                ))
        else:
            raise NotImplementedError()

        schedulers = []
        for optimizer in optimizers:
            if self.lr_sched.lower() == 'multistep':
                schedulers.append(lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma']))
            elif self.lr_sched.lower() == 'cosine':
                schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max']))
            elif self.lr_sched.lower() == 'linear':
                schedulers.append(lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_sched_args['start_factor'],
                    end_factor=self.lr_sched_args['end_factor'],
                    total_iters=self.lr_sched_args['total_iters']
                ))
            

        return optimizers, schedulers
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, Pred, Thermal, RGB, loss_type=None):
        if loss_type == None:
            loss = self.loss_fn(Pred, Thermal)
        elif loss_type == "G_pix2pix":
            fake_AB = torch.cat((RGB, Pred), 1)
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, True)
            loss += self.loss_config["G_loss_lambda"] * self.loss_fn_L1(Pred, Thermal)
        elif loss_type == "D_pix2pix" or loss_type == "D_pix2pixHD":
            fake_AB = torch.cat((RGB, Pred.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, False)
            real_AB = torch.cat((RGB, Thermal), 1)
            pred_real = self.discriminator(real_AB)
            loss += self.loss_fn_GAN(pred_real, True)
            loss = 0.5 * loss
        elif loss_type == "G_pix2pixHD":
            fake_AB = torch.cat((RGB, Pred), 1)
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, True)
            real_AB = torch.cat((RGB, Thermal), 1)
            pred_real = self.discriminator(real_AB)
            feat_weights = 4.0 / (self.model_config["n_layers_D"] + 1)
            D_weights = 1.0 / self.model_config["num_D"]
            loss_G_GAN_Feat = 0
            for i in range(self.model_config["num_D"]):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.loss_fn_L1(pred_fake[i][j], pred_real[i][j].detach()) * self.loss_config["G_loss_lambda"]
            loss += loss_G_GAN_Feat
        elif loss_type == "G_cyclegan":
            Pred_Thermal, Rec_RGB, Rec_Thermal, Pred_RGB = Pred
            loss = self.loss_fn_GAN(self.discriminator_B(Pred_Thermal), True)
            loss += self.loss_fn_GAN(self.discriminator_A(Pred_RGB), True)
            loss += self.loss_config["G_loss_lambda_Thermal"] * self.loss_fn_L1(Rec_Thermal, Thermal)
            loss += self.loss_config["G_loss_lambda_RGB"] * self.loss_fn_L1(Rec_RGB, RGB)
        elif loss_type == "D_cyclegan":
            Pred_Thermal, Rec_RGB, Rec_Thermal, Pred_RGB = Pred
            loss = self.loss_fn_GAN(self.discriminator_B(Pred_Thermal.detach()), False)
            loss += self.loss_fn_GAN(self.discriminator_B(Thermal), True)
            loss += self.loss_fn_GAN(self.discriminator_A(Pred_RGB.detach()), False)
            loss += self.loss_fn_GAN(self.discriminator_A(RGB), True)
            loss = 0.5 * loss
        elif loss_type == "G_vqgan":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, cond=RGB, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_vqgan":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, cond=RGB, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_klvae":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_klvae":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_klvae_RGB":
            Pred_RGB, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, RGB, Pred_RGB, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_klvae_RGB":
            Pred_RGB, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, RGB, Pred_RGB, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_dcae":
            Pred_Thermal = Pred
            if hasattr(self.model.decoder.conv_out, "weight"):
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            else:
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.conv.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_dcae":
            Pred_Thermal = Pred
            if hasattr(self.model.decoder.conv_out, "weight"):
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            else:
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.conv.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "Diff_sit":
            loss = Pred.mean()
        else:
            raise NotImplementedError(f"{loss_type} not found")
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx, vis_num=4):
        RGB_list = batch[0]
        Thermal_list = batch[1]
        dataset_idx_list = batch[2]

        thermal_input = self.model_arch == "cyclegan" or self.model_arch == "klvae" or self.model_arch == "dcae" or self.model_arch == "sit"
        Pred_list = self(RGB_list, dataset_idx_list, Thermal=Thermal_list if thermal_input else None, Training=True)

        find_nan = False
        if type(Pred_list) != list:
            if torch.isnan(Pred_list).any():
                print('NaNs in Pred_list. Skip it')
                find_nan = True
        elif type(Pred_list) == list:
            for item in Pred_list:
                if torch.isnan(item).any():
                    print('NaNs in Pred_list. Skip it')
                    find_nan = True
        if find_nan:
            return {}

        # Upload pred and GT for vis
        if not self.log_img_first_iter_train:
            RGB_list_vis = RGB_list.cpu() * 0.5 + 0.5
            RGB_list_vis = torch.clamp(RGB_list_vis, 0, 1)
            Thermal_list_vis = Thermal_list.cpu() * 0.5 + 0.5
            Thermal_list_vis = torch.clamp(Thermal_list_vis, 0, 1)
            if type(Pred_list)!=list:
                Pred_list_vis = Pred_list.cpu() * 0.5 + 0.5
                Pred_list_vis = torch.clamp(Pred_list_vis, 0, 1)
            else:
                Pred_list_vis = Pred_list[0].cpu() * 0.5 + 0.5
                Pred_list_vis = torch.clamp(Pred_list_vis, 0, 1)
            self.vis_eval_image(RGB_list_vis, Thermal_list_vis, Pred_list_vis, vis_num, "mixed", 'train')

        if self.model_arch == 'pix2pix' or \
           self.model_arch == "cyclegan" or \
           self.model_arch == "pix2pixHD" or \
           self.model_arch == "vqgan" or \
           self.model_arch == "klvae" or \
           self.model_arch == "klvae_RGB" or \
           self.model_arch == "dcae":
            # GAN training
            # Train G
            if not ((self.training_stage == "full" or self.training_stage == "mid") and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae")):
                opt_g, opt_d = self.optimizers()
            else:
                opt_g = self.optimizers()
            self.toggle_optimizer(opt_g)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"G_{self.model_arch}")
            self.log('loss_G', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_g.step()
                opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)

            # Train D
            # DCAE and KLVAE does not use D for full and mid
            if not ((self.training_stage == "full" or self.training_stage == "mid") and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae")):
                self.toggle_optimizer(opt_d)
                loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"D_{self.model_arch}")
                self.log('loss_D', loss.item(), logger=True, prog_bar=True, sync_dist=True)
                self.manual_backward(loss)
                if batch_idx % self.gradient_accumulation == 0:
                    opt_d.step()
                    opt_d.zero_grad()
                self.untoggle_optimizer(opt_d)
        elif self.model_arch == "sit":
            opt_diff = self.optimizers()
            self.toggle_optimizer(opt_diff)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"Diff_{self.model_arch}")
            self.log('loss_Diff', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_diff.step()
                opt_diff.zero_grad()
            self.update_ema(self.ema, self.model)
            self.untoggle_optimizer(opt_diff)
        else:
            raise NotImplementedError()

        self.log_img_first_iter_train = True

        return {'loss': loss}
    
    def calc_loss_batch(self, Pred_list, Thermal_list, RGB_list, loss_type=None):
        loss = self.loss_function(Pred_list, Thermal_list, RGB_list, loss_type)
        return loss

    def on_train_epoch_start(self):
        self.log_img_first_iter_train = False
        if "klvae" in self.model_arch or "dcae" in self.model_arch or ("sit" in self.model_arch and self.calculate_stats):
            self.latent_list = torch.empty(0)
            if "sit" in self.model_arch:
                self.latent_RGB_list = torch.empty(0)
        if self.model_arch == "sit" and hasattr(self, 'latent_cache') and self.current_epoch % self.model_config['cache_rate'] == 0:
            self.latent_cache = []
            if not self.RGB_encoder_training:
                self.RGB_latent_cache = []
            else:
                self.RGB_cache = []
            self.latent_idx = 0
        elif self.model_arch == "sit" and hasattr(self, 'latent_cache'):
            perm = torch.randperm(len(self.latent_cache))
            self.latent_cache =  [self.latent_cache[i] for i in perm]
            if not self.RGB_encoder_training:
                self.RGB_latent_cache = [self.RGB_latent_cache[i] for i in perm]
            else:
                self.RGB_cache = [self.RGB_cache[i] for i in perm]
            self.latent_idx = 0

    def on_train_epoch_end(self):
        self.log_img_first_iter_train = True
        if "klvae" in self.model_arch or "dcae" in self.model_arch or ("sit" in self.model_arch and self.calculate_stats):
            self.latent_std = self.latent_list.std().item()
            self.latent_mean = self.latent_list.mean().item()
            print(f"Latent Standard Deviation: {self.latent_std}")
            print(f"Latent Mean: {self.latent_mean}")
            self.log("latent_std", self.latent_std, sync_dist=True)
            self.log("latent_mean", self.latent_mean, sync_dist=True)
            self.log("latent_normalizer", 1 / self.latent_std, sync_dist=True)
            if "sit" in self.model_arch:
                self.latent_RGB_std = self.latent_RGB_list.std(dim=[0, 2, 3]).tolist() # Needed for batch norm init
                self.latent_RGB_mean = self.latent_RGB_list.mean(dim=[0, 2, 3]).tolist() # Needed for batch norm init
                print(f"RGB Latent Standard Deviation: {self.latent_RGB_std}")
                print(f"RGB Latent Mean: {self.latent_RGB_mean}")
                self.log_dict({f"RGB_latent_std_{i}": self.latent_RGB_std[i] for i in range(len(self.latent_RGB_std))})
                self.log_dict({f"RGB_latent_mean_{i}": self.latent_RGB_mean[i] for i in range(len(self.latent_RGB_std))})
        try:
            if self.lr_schedulers() is not None:
                for lr_scheduler in self.lr_schedulers():
                    lr_scheduler.step()
        except:
            self.lr_schedulers().step()
        if hasattr(self, 'latent_cache_init') and self.latent_cache_init == False:
            self.latent_cache_init = True
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None, vis_num=4):
        RGB, Thermal, dataset_idx = batch
        if self.model_arch == "klvae" or self.model_arch == "dcae":
            output = self(RGB, dataset_idx, Thermal=Thermal)
        else:
            output = self(RGB, dataset_idx)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.eval_calculate_metrics()
            self.eval_outputs = []
            self.current_dataloader_idx = dataloader_idx
            self.log_img_first_iter_val = False
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        Thermal = Thermal * 0.5 + 0.5
        Thermal = torch.clamp(Thermal, 0, 1)
        if self.model_arch == "klvae_RGB":
            self.eval_outputs.append((output.detach().cpu(), RGB))
        else:
            self.eval_outputs.append((output.detach().cpu(), Thermal))

        if not self.log_img_first_iter_val:
            eval_dataset_name = self.trainer.datamodule.val_datasets[dataloader_idx].dataset_name
            self.vis_eval_image(RGB, Thermal, output, vis_num, eval_dataset_name, 'val')

        self.log_img_first_iter_val = True

        return output.detach().cpu(), Thermal
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.eval_outputs = []
        self.results_list = []
        self.current_dataloader_idx = 0
        self.log_img_first_iter_val = False
    
    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        self.eval_calculate_metrics() # For last dataset
        for i, eval_dataset in enumerate(dm.val_datasets):
            eval_set_name = eval_dataset.dataset_name
            results_dict = self.results_list[i]
            if results_dict == []:
                continue
            self.log(f'{eval_set_name}_{eval_dataset.split}/PSNR', results_dict['PSNR'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/SSIM', results_dict['SSIM'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/FID', results_dict['FID'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/LPIPS', results_dict['LPIPS'], prog_bar=False, logger=True)        
        print('\n\n')
        # reset the outputs list
        self.eval_outputs = []
        self.results_list = []
        self.log_img_first_iter_val = False

    def test_step(self, batch, batch_idx, dataloader_idx=None, vis_num=4):
        RGB, Thermal, dataset_idx = batch
        if self.model_arch == "klvae" or self.model_arch == "dcae":
            output = self(RGB, dataset_idx, Thermal=Thermal)
        else:
            output = self(RGB, dataset_idx)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.eval_calculate_metrics()
            self.eval_outputs = []
            self.current_dataloader_idx = dataloader_idx
            self.log_img_first_iter_test = False
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        Thermal = Thermal * 0.5 + 0.5
        Thermal = torch.clamp(Thermal, 0, 1)
        if self.model_arch == "klvae_RGB":
            self.eval_outputs.append((output.detach().cpu(), RGB))
        else:
            self.eval_outputs.append((output.detach().cpu(), Thermal))

        if not self.log_img_first_iter_test:
            eval_dataset_name = self.trainer.datamodule.test_datasets[dataloader_idx].dataset_name
            self.vis_eval_image(RGB, Thermal, output, vis_num, eval_dataset_name, 'test')

        self.log_img_first_iter_test = True

        return output.detach().cpu(), Thermal
    
    def prediction_step(self, batch, batch_idx):
        # Only for SiT generating map
        RGB, coordinates, dataset_idx = batch
        output = self(RGB, dataset_idx)
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        return output.detach().cpu(), coordinates
    
    def on_test_epoch_start(self):
        # reset the outputs list
        self.eval_outputs = []
        self.results_list = []
        self.current_dataloader_idx = 0
        self.log_img_first_iter_test = False
    
    def on_test_epoch_end(self):
        dm = self.trainer.datamodule
        self.eval_calculate_metrics() # For last dataset
        for i, eval_dataset in enumerate(dm.test_datasets):
            eval_set_name = eval_dataset.dataset_name
            results_dict = self.results_list[i]
            if results_dict == []:
                continue
            self.log(f'{eval_set_name}_{eval_dataset.split}/PSNR', results_dict['PSNR'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/SSIM', results_dict['SSIM'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/FID', results_dict['FID'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/LPIPS', results_dict['LPIPS'], prog_bar=False, logger=True)        
        print('\n\n')
        # reset the outputs list
        self.eval_outputs = []
        self.results_list = []
        self.log_img_first_iter_test = False

    def eval_calculate_metrics(self):
        # Calculate evaluation metrics and save to results_list
        psnr = None
        ssim = None
        fid = None
        lpips = None
        for Pred, Thermal in tqdm(self.eval_outputs, total=len(self.eval_outputs)):
            Pred = Pred.to("cuda") if torch.cuda.is_available() else Pred
            Thermal = Thermal.to("cuda") if torch.cuda.is_available() else Thermal
            psnr = calculate_psnr(Pred, Thermal, psnr)
            ssim = calculate_ssim(Pred, Thermal, ssim)
            fid = calculate_fid(Pred, Thermal, fid)
            lpips = calculate_lpips(Pred, Thermal, lpips)
        mean_PSNR = psnr.compute()
        mean_SSIM = ssim.compute()
        mean_FID = fid.compute()
        mean_LPIPS = lpips.compute()
        results_dict = {'PSNR': mean_PSNR, 'SSIM': mean_SSIM, "FID": mean_FID, "LPIPS": mean_LPIPS}
        self.results_list.append(results_dict)

    def vis_eval_image(self, RGB, Thermal, output, vis_num, eval_dataset_name, split, image_norm="normal"):
        if image_norm == "normal":
            image_mean_std = NORMAL_MEAN_STD
        elif image_norm == "imagenet":
            image_mean_std = IMAGENET_MEAN_STD
        else:
            raise NotImplementedError()
        denormalized_image = RGB.cpu()
        list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
        list_images = list_images[:vis_num]
        self.logger.log_image(f'input_{split}_images_{eval_dataset_name}', list_images)
        denormalized_image = Thermal.cpu()
        list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
        list_images = list_images[:vis_num]
        self.logger.log_image(f'gt_{split}_thermal_{eval_dataset_name}', list_images)
        if not (self.model_arch == "sit" and split == "train"):
            denormalized_image = output.cpu()
            list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
            list_images = list_images[:vis_num]
            self.logger.log_image(f'pred_{split}_thermal_{eval_dataset_name}', list_images)

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

        ema_buffers = OrderedDict(ema_model.named_buffers())
        model_buffers = OrderedDict(model.named_buffers())

        for name, buffer in model_buffers.items():
            name = name.replace("module.", "")
            if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                # Apply EMA only to float buffers
                ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
            else:
                # Direct copy for non-float buffers
                ema_buffers[name].copy_(buffer)

    def load_pretrained(self, model):
        state_dict = torch.load(self.model_config['vae_path'])['state_dict']
        new_state_dict = {}
        for old_key, value in state_dict.items():
            # 1) Skip any keys you definitely don’t need:
            if old_key.startswith("loss_fn."):
                continue
            
            # 2) Strip off "model." if the model was saved that way:
            if old_key.startswith("model."):
                new_key = old_key.replace("model.", "")  # remove the "model." prefix
            else:
                new_key = old_key
            
            # Now add to the new dict
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        return model
    
    def pad_to_divisble(self, input, multiple=8):
        batch, channels, height, width = input.shape
        padded_height = ((height + multiple - 1) // multiple) * multiple
        padded_width = ((width + multiple - 1) // multiple) * multiple
        
        pad_top = 0
        pad_bottom = padded_height - height
        pad_left = 0
        pad_right = padded_width - width
        
        padded_image = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_image
    
    def preprocess_raw_image(self, x, enc_type):
        resolution = x.shape[-1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # For thermal
        if 'dinov2' in enc_type:
            x = x * 0.5 + 0.5
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        return x
    
    def load_encoder(self, enc_type):
        resolution = 256
        if 'dinov2' in enc_type:
            import timm
            if 'reg' in enc_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder.eval()
        return encoder