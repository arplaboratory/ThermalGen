import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import to_pil_image
from diffusers.models import AutoencoderKL
from models.generative_models.sit_networks import sit_networks
from models.generative_models.sit_networks.transport import create_transport, Sampler
import yaml
import os
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin

class ThermalGenSIT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_config):
        
        super().__init__()
        self.model_config = model_config
        num_classes = 1000

        # --- Build SiT backbone (KL-VAE only)
        assert model_config['vae_model'] == "klvae", "Only klvae is supported"
        self.model = sit_networks.SiT_models[
            f"SiT-{model_config['arch']}/{model_config['patch_size']}"
        ](
            in_channels=4,
            num_classes=num_classes,
            injection_args=model_config['injection_args'],
            learn_sigma=False,
        )

        # Thermal + RGB VAEs
        self.thermal_vae = AutoencoderKL(**model_config['vae_config'])
        self.RGB_vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{model_config['vae']}"
        )

        # Transport & Sampler
        self.transport = create_transport(**model_config['transport_config'])
        self.sampler = Sampler(self.transport)

        # EMA copy for eval
        self.ema = self.model
        self.use_cfg = model_config['cfg_scale'] > 1.0
        self.thermal_normalizer = model_config.get('thermal_normalizer', None)
        self.RGB_normalizer = model_config.get('RGB_normalizer', None)

    @torch.no_grad()
    def forward(self, RGB, dataset_idx):
        # latent sampling size is tied to VAE downsample ratio (8 for KL-VAE)
        latent_size = RGB.shape[2] // 8, RGB.shape[3] // 8
        zs = torch.randn(RGB.shape[0], 4, *latent_size, device=RGB.device)

        # Encode RGB with KL-VAE
        x_RGB = self.RGB_vae.encode(RGB).latent_dist.sample()
        if self.RGB_normalizer is not None:
            x_RGB = x_RGB * self.RGB_normalizer

        # Conditional sampling
        ys = dataset_idx
        sample_fn = self.sampler.sample_ode()
        if self.use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([1000] * len(ys), device=RGB.device)
            ys = torch.cat([ys, y_null], 0)
            x_RGB = torch.cat([x_RGB, x_RGB], 0)
            model_eval = self.ema.forward_with_cfg
            kwargs = dict(y=ys, x_RGB=x_RGB, cfg_scale=self.model_config['cfg_scale'])
        else:
            model_eval = self.ema.forward
            kwargs = dict(y=ys, x_RGB=x_RGB)

        samples = sample_fn(zs, model_eval, **kwargs)[-1]
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # Decode thermal latent
        if self.thermal_normalizer is not None:
            samples = samples / self.thermal_normalizer
        Pred_Thermal = self.thermal_vae.decode(samples).sample

        return Pred_Thermal[:, :, :RGB.shape[2], :RGB.shape[3]]
            
if __name__ == "__main__":

    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # eval transform
    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    model = ThermalGenSIT.from_pretrained("xjh19972/ThermalGen-XL-2").to(device)

    # -----------------------------
    # Inputs
    # -----------------------------
    RGB_image = Image.open("test_color.png")
    RGB = eval_transform(RGB_image).unsqueeze(0).to(device)

    # -----------------------------
    # Inference - Unconditional
    # -----------------------------
    with torch.no_grad():
        dataset_idx = torch.ones(1, dtype=torch.long, device=device) * 1000  # class labels
        pred_thermal = model(RGB, dataset_idx)
        # normalize to [0,1]
        pred_thermal = pred_thermal * 0.5 + 0.5
        pred_thermal = torch.clamp(pred_thermal, 0, 1)

    thermal_img = to_pil_image(pred_thermal[0].cpu())
    thermal_img.save("test_thermal_uncond.png")
    print("Saved predicted thermal image to test_thermal_uncond.png")

    # -----------------------------
    # Inference - Conditional
    # -----------------------------
    with torch.no_grad():
        dataset_idx = torch.ones(1, dtype=torch.long, device=device) * 7  # class labels for M3FD (Refer to configs/datasets)
        pred_thermal = model(RGB, dataset_idx)
        # normalize to [0,1]
        pred_thermal = pred_thermal * 0.5 + 0.5
        pred_thermal = torch.clamp(pred_thermal, 0, 1)

    thermal_img = to_pil_image(pred_thermal[0].cpu())
    thermal_img.save("test_thermal_cond.png")
    print("Saved predicted thermal image to test_thermal_cond.png")

    # -----------------------------
    # Inference - Conditional - CFG = 4.0
    # -----------------------------
    with torch.no_grad():
        dataset_idx = torch.ones(1, dtype=torch.long, device=device) * 7  # class labels for M3FD (Refer to configs/datasets)
        model.use_cfg = True
        model.model_config["cfg_scale"] = 4.0
        pred_thermal = model(RGB, dataset_idx)
        # normalize to [0,1]
        pred_thermal = pred_thermal * 0.5 + 0.5
        pred_thermal = torch.clamp(pred_thermal, 0, 1)

    thermal_img = to_pil_image(pred_thermal[0].cpu())
    thermal_img.save("test_thermal_cond_cfg4.png")
    print("Saved predicted thermal image to test_thermal_cond_cfg4.png")
