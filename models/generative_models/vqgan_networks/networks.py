import torch
from models.generative_models.vqgan_networks.diffusionmodules import Encoder, Decoder
from models.generative_models.vqgan_networks.quantize import VectorQuantizer2 as VectorQuantizer

class VQGAN(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):   
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(ddconfig["n_embed"], ddconfig["embed_dim"], beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["embed_dim"], ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight