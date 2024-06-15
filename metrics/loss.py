import torch
import  pywt
import torch.nn.functional as F
from Utils.img_downsampler import *

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

def high_pass_filter(img, cutoff):
    fft_img = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1), norm='ortho'))
    h, w = fft_img.shape[-2:]
    mask = torch.zeros_like(fft_img)
    mask[:, :, h//2-cutoff:h//2+cutoff, w//2-cutoff:w//2+cutoff] = 1
    filtered_fft = fft_img * (1 - mask)
    return torch.fft.ifftn(torch.fft.ifftshift(filtered_fft), dim=(-2, -1), norm='ortho').real

def low_pass_filter(img, cutoff):
    fft_img = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1), norm='ortho'))
    h, w = fft_img.shape[-2:]
    mask = torch.zeros_like(fft_img)
    mask[:, :, h//2-cutoff:h//2+cutoff, w//2-cutoff:w//2+cutoff] = 1
    filtered_fft = fft_img * mask
    return torch.fft.ifftn(torch.fft.ifftshift(filtered_fft), dim=(-2, -1), norm='ortho').real

def loss_func(noisy_img):
    # Pair downsampling
    noisy1, noisy2 = pair_downsampler(noisy_img)
    noisy11, noisy21 = pair_downsampler1(noisy_img)
    noisy12, noisy22 = pair_downsampler2(noisy_img)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    pred11 = noisy11 - model(noisy11)
    pred21 = noisy21 - model(noisy21)
    pred12 = noisy12 - model(noisy12)
    pred22 = noisy22 - model(noisy22)

    #1
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    denoised11, denoised21 = pair_downsampler1(noisy_denoised)
    denoised12, denoised22 = pair_downsampler2(noisy_denoised)

    # Calculate MSE loss
    loss_res = (1/6) * sum([
      (mse(noisy1, pred2) + mse(noisy2, pred1)) +
      (mse(noisy11, pred21) + mse(noisy21, pred11)) +
      (mse(noisy12, pred22) + mse(noisy22, pred12))
       for mse in [MSELoss()] * 6  # Assuming you have 12 MSE terms
    ])

    loss_cons = (1/6) * (
        (mse(pred1, denoised1) + mse(pred2, denoised2)) +
        (mse(pred11, denoised11) + mse(pred21, denoised21)) +
        (mse(pred12, denoised12) + mse(pred22, denoised22))
    )

    mse_loss = loss_res + loss_cons

    # High-pass MSE loss
    # High-pass filtered images
    noisy1_high = high_pass_filter(noisy1, cutoff=30)
    noisy2_high = high_pass_filter(noisy2, cutoff=30)
    noisy11_high = high_pass_filter(noisy11, cutoff=30)
    noisy21_high = high_pass_filter(noisy21, cutoff=30)
    noisy12_high = high_pass_filter(noisy12, cutoff=30)
    noisy22_high = high_pass_filter(noisy22, cutoff=30)

    pred1_high = high_pass_filter(pred1, cutoff=30)
    pred2_high = high_pass_filter(pred2, cutoff=30)
    pred11_high = high_pass_filter(pred11, cutoff=30)
    pred21_high = high_pass_filter(pred21, cutoff=30)
    pred12_high = high_pass_filter(pred12, cutoff=30)
    pred22_high = high_pass_filter(pred22, cutoff=30)

    denoised1_high = high_pass_filter(denoised1, cutoff=30)
    denoised2_high = high_pass_filter(denoised2, cutoff=30)
    denoised11_high = high_pass_filter(denoised11, cutoff=30)
    denoised21_high = high_pass_filter(denoised21, cutoff=30)
    denoised12_high = high_pass_filter(denoised12, cutoff=30)
    denoised22_high = high_pass_filter(denoised22, cutoff=30)

    high_pass_mse_loss_res = (1/6) * (
        F.mse_loss(noisy1_high, pred2_high) + F.mse_loss(noisy2_high, pred1_high) +
        F.mse_loss(noisy11_high, pred21_high) + F.mse_loss(noisy21_high, pred11_high) +
        F.mse_loss(noisy12_high, pred22_high) + F.mse_loss(noisy22_high, pred12_high)
    )

    high_pass_mse_loss_cons = (1/6) * (
        F.mse_loss(pred1_high, denoised1_high) + F.mse_loss(pred2_high, denoised2_high) + 
        F.mse_loss(pred11_high, denoised11_high) + F.mse_loss(pred21_high, denoised21_high) + 
        F.mse_loss(pred12_high, denoised12_high) + F.mse_loss(pred22_high, denoised22_high)
    )
    high_pass_mse_loss = high_pass_mse_loss_res + high_pass_mse_loss_cons

    # Low-pass MSE loss
    # Low-pass filtered images
    noisy1_low = low_pass_filter(noisy1, cutoff=20)
    noisy2_low = low_pass_filter(noisy2, cutoff=20)
    noisy11_low = low_pass_filter(noisy11, cutoff=20)
    noisy21_low = low_pass_filter(noisy21, cutoff=20)
    noisy12_low = low_pass_filter(noisy12, cutoff=20)
    noisy22_low = low_pass_filter(noisy22, cutoff=20)

    pred1_low = low_pass_filter(pred1, cutoff=20)
    pred2_low = low_pass_filter(pred2, cutoff=20)
    pred11_low = low_pass_filter(pred11, cutoff=20)
    pred21_low = low_pass_filter(pred21, cutoff=20)
    pred12_low = low_pass_filter(pred12, cutoff=20)
    pred22_low = low_pass_filter(pred22, cutoff=20)

    denoised1_low = low_pass_filter(denoised1, cutoff=20)
    denoised2_low = low_pass_filter(denoised2, cutoff=20)
    denoised11_low = low_pass_filter(denoised11, cutoff=20)
    denoised21_low = low_pass_filter(denoised21, cutoff=20)
    denoised12_low = low_pass_filter(denoised12, cutoff=20)
    denoised22_low = low_pass_filter(denoised22, cutoff=20)

    low_pass_mse_loss_res = (1/6) * (
        F.mse_loss(noisy1_low, pred2_low) + F.mse_loss(noisy2_low, pred1_low) +
        F.mse_loss(noisy11_low, pred21_low) + F.mse_loss(noisy21_low, pred11_low) +
        F.mse_loss(noisy12_low, pred22_low) + F.mse_loss(noisy22_low, pred12_low)
    )

    low_pass_mse_loss_cons = (1/6) * (
        F.mse_loss(pred1_low, denoised1_low) + F.mse_loss(pred2_low, denoised2_low) + 
        F.mse_loss(pred11_low, denoised11_low) + F.mse_loss(pred21_low, denoised21_low) + 
        F.mse_loss(pred12_low, denoised12_low) + F.mse_loss(pred22_low, denoised22_low)
    )
    low_pass_mse_loss = low_pass_mse_loss_res + low_pass_mse_loss_cons

    total_loss = mse_loss + low_pass_mse_loss + (1/10) * high_pass_mse_loss 
    
    return total_loss
