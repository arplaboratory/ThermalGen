from models.generative_models.pix2pix_networks.networks import GANLoss
from models.generative_models.vqgan_networks.vqlpips import VQLPIPSWithDiscriminator, LPIPSWithDiscriminator, DCAELPIPSWithDiscriminator
from models.generative_models.pix2pixHD_networks.networks import GANLoss_multi

import torch

def get_loss(loss_name, loss_config):
    if loss_name == "pix2pix" or loss_name == "cyclegan":
        loss_GAN = GANLoss(loss_config["GAN_mode"])
        loss_L1 = torch.nn.L1Loss()
        return loss_GAN, loss_L1
    elif loss_name == "pix2pixHD":
        loss_GAN = GANLoss_multi(use_lsgan = (loss_config["GAN_mode"] == 'lsgan'))
        loss_L1 = torch.nn.L1Loss()
        return loss_GAN, loss_L1
    elif loss_name == "vqgan":
        loss_lpips = VQLPIPSWithDiscriminator(**loss_config)
        return loss_lpips
    elif loss_name == "klvae":
        loss_lpips = LPIPSWithDiscriminator(**loss_config)
        return loss_lpips
    elif loss_name == "dcae":
        loss_lpips = DCAELPIPSWithDiscriminator(**loss_config)
        return loss_lpips
    elif loss_name == "sit":
        return None
    else:
        raise NotImplementedError()