# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from timm.layers import use_fused_attn
from timm.layers.helpers import to_2tuple
from timm.models.vision_transformer import Attention, Mlp
import torch.nn.functional as F

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class PatchEmbedFlexible(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            injection_position: str = "kv",
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.injection_position = injection_position

        if self.injection_position == "kv" or self.injection_position == "qk" or self.injection_position == "qv":
            self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
            self.qkv_RGB = nn.Linear(dim, dim * 2, bias=qkv_bias)
        elif self.injection_position == "q" or self.injection_position == "k" or self.injection_position == "v":
            self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.qkv_RGB = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            raise NotImplementedError()
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, x_RGB: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.injection_position == "kv": # Traditional
            q = self.qkv(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            kv = self.qkv_RGB(x_RGB).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q = q.unbind(0)[0]
            k, v = kv.unbind(0)
        elif self.injection_position == "qk":
            v = self.qkv(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            qk = self.qkv_RGB(x_RGB).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            v = v.unbind(0)[0]
            q, k = qk.unbind(0)
        elif self.injection_position == "qv":
            k = self.qkv(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            qv = self.qkv_RGB(x_RGB).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k = k.unbind(0)[0]
            q, v = qv.unbind(0)
        elif self.injection_position == "q":
            kv = self.qkv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q = self.qkv_RGB(x_RGB).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q = q.unbind(0)[0]
            k, v = kv.unbind(0)
        elif self.injection_position == "k":
            qv = self.qkv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k = self.qkv_RGB(x_RGB).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k = k.unbind(0)[0]
            q, v = qv.unbind(0)
        elif self.injection_position == "v":
            qk = self.qkv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            v = self.qkv_RGB(x_RGB).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            v = v.unbind(0)[0]
            q, k = qk.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, 
                 hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 injection_args=None,
                 **block_kwargs):
        super().__init__()
        self.injection_method = injection_args["injection_method"]
        if self.injection_method == "cross":
            self.self_attn = injection_args["self_attn"] if "self_attn" in injection_args else True
            self.first_cross = injection_args["first_cross"] if "first_cross" in injection_args else False
            if self.first_cross:
                assert self.self_attn
            self.injection_position = injection_args["injection_position"]
            if self.self_attn:
                self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.cross_attn= CrossAttentionBlock(hidden_size, num_heads=num_heads, qkv_bias=True, injection_position=self.injection_position, **block_kwargs)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm1c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            if self.self_attn:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 9 * hidden_size, bias=True)
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=True)
                )
        elif self.injection_method == "concat":
            self.replace_RGB = injection_args["replace_RGB"] if "replace_RGB" in injection_args else False
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            raise NotImplementedError()

    def forward(self, x, c, x_RGB, last_block=False, first_block=False):
        if self.injection_method == "cross":
            if self.self_attn:
                shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
                if self.first_cross:
                    x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm1c(x), shift_mca, scale_mca), self.norm2c(x_RGB))
                    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                else:
                    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                    x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm1c(x), shift_mca, scale_mca), self.norm2c(x_RGB))
            else:
                shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm1c(x), shift_mca, scale_mca), self.norm2c(x_RGB))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        elif self.injection_method == "concat":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            if first_block:
                x = torch.cat([x, x_RGB], dim=1)
            elif self.replace_RGB:
                x[:, x.shape[1]//2:, :] = x_RGB
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            if last_block:
                x, _ = x.chunk(2, dim=1)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        injection_args=None,
        repa=False,
        repa_encoder_depth=8,
        repa_z_dims=[768],
        repa_projector_dim=2048,
        repa_full_cover=False,
        **block_kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.injection_args = injection_args
        if injection_args is not None:
            self.rgb_dropout_prob = injection_args['rgb_dropout_prob'] if 'rgb_dropout_prob' in injection_args else 0.0
        else:
            self.rgb_dropout_prob = 0.0
        self.repa = repa
        self.repa_encoder_depth = repa_encoder_depth
        self.repa_full_cover = repa_full_cover

        self.x_embedder = PatchEmbedFlexible(patch_size, in_channels, hidden_size, bias=True)
        self.x_RGB_embedder = PatchEmbedFlexible(patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, injection_args=injection_args, **block_kwargs) for _ in range(depth)
        ])
        if self.repa:
            if self.injection_args['injection_method'] == "concat" and self.repa_full_cover:
                self.projectors = nn.ModuleList([
                    build_mlp(hidden_size * 2, repa_projector_dim, z_dim) for z_dim in repa_z_dims
                    ])
            else:
                self.projectors = nn.ModuleList([
                    build_mlp(hidden_size, repa_projector_dim, z_dim) for z_dim in repa_z_dims
                    ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        if 'bn_momentum' in injection_args and injection_args['bn_momentum'] > 0:
            self.bn_RGB = torch.nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=injection_args['bn_momentum'], affine=False, track_running_stats=True
            )
            self.init_bn(torch.tensor(injection_args['bn_RGB_init_std']), torch.tensor(injection_args['bn_RGB_init_bias']))
        else:
            self.bn_RGB = None

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def init_bn(self, latents_RGB_std, latents_RGB_bias):
        self.bn_RGB.running_mean = latents_RGB_bias
        self.bn_RGB.running_var = latents_RGB_std.pow(2)

    def unpatchify(self, x, num_patches):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = num_patches[0]
        w = num_patches[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, x_RGB):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        num_patches = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, num_patches[0], num_patches[1]).to(x.device).to(x.dtype)
        rgb_pos_embed = pos_embed
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        if self.training and self.rgb_dropout_prob > 0.0:
            drop_ids = torch.rand(x_RGB.shape[0], device=x_RGB.device) < self.rgb_dropout_prob
            x_RGB[drop_ids] = 0.0
        if self.bn_RGB is not None:
            x_RGB = self.bn_RGB(x_RGB)
        x_RGB = self.x_RGB_embedder(x_RGB) + rgb_pos_embed
        c = t + y                                # (N, D)
        for i, block in enumerate(self.blocks):
            x = block(x, c, x_RGB, last_block=True if i == len(self.blocks) - 1 else False, first_block=True if i == 0 else False)  # (N, T, D)
            if self.repa and self.training and (i + 1) == self.repa_encoder_depth:
                if self.injection_args['injection_method'] == "concat":
                    if self.repa_full_cover:
                        x_input = x
                    else:
                        x_input, _ = x.chunk(2, dim=1)
                else:
                    x_input = x
                zs = [projector(x_input.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, num_patches)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        if self.repa and self.training:
            return x, zs
        else:
            return x

    def forward_with_cfg(self, x, t, y, x_RGB, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, x_RGB)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
    grid = torch.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concatenate([torch.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}