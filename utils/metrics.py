from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

def calculate_psnr(pred, target, psnr):
    if psnr is None:
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    psnr.update(pred, target)
    return psnr

def calculate_ssim(pred, target, ssim):
    if ssim is None:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim.update(pred, target)
    return ssim

def calculate_fid(pred, target, fid):
    if fid is None:
        fid = FrechetInceptionDistance(normalize=True).to(pred.device)
    with torch.autocast("cuda", torch.bfloat16 if pred.dtype == torch.bfloat16 else torch.float32):
        if pred.shape[1] == 1:
            fid.update(pred.repeat(1, 3, 1, 1), real=False)
            fid.update(target.repeat(1, 3, 1, 1), real=True)
    return fid

def calculate_lpips(pred, target, lpips, net_type="alex"):
    if lpips is None:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(pred.device)
    if pred.shape[1] == 1:
        lpips.update(pred.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1))
    else:
        lpips.update(pred, target)
    return lpips