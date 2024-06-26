import torch
import  pywt
import torch.nn.functional as F
from Utils.img_downsampler import *
def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


def loss_func(noisy_img , model, method_type='modified1'):
    if method_type == 'modified':
        noisy1, noisy2 = pair_downsampler1(noisy_img)
    else:
        noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    if method_type == 'modified1':
        denoised1, denoised2 = pair_downsampler1(noisy_denoised)
    else:
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss

def loss_func13(noisy_img , model, method_type):
    if method_type == 'modified':
        noisy1, noisy2 = pair_downsampler1(noisy_img)
        noisy11, noisy22 = pair_downsampler2(noisy_img)
    else:
        noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    pred11 = noisy11 - model(noisy11)
    pred22 = noisy22 - model(noisy22)

    loss_res1 = 1 / 2 * (mse(noisy11, pred22) + mse(noisy22, pred11))
    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    if method_type == 'modified':
        denoised1, denoised2 = pair_downsampler1(noisy_denoised)
        denoised11, denoised22 = pair_downsampler2(noisy_denoised)
    else:
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons1 = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))
    loss_cons = 1 / 2 * (mse(pred11, denoised11) + mse(pred22, denoised22))

    grad_x = torch.abs(denoised1[:, :, :, :-1] - denoised1[:, :, :, 1:])
    grad_y = torch.abs(denoised1[:, :, :-1, :] - denoised1[:, :, 1:, :])
    tv_loss = torch.mean(grad_x) + torch.mean(grad_y)

    loss_cons_total = (loss_cons1 + loss_cons)/2
    loss_res_total = (loss_res + loss_res1)/2
    #loss = loss_res + loss_cons + 0.004 * tv_loss
    #loss = 0.8 * (loss_res + loss_cons) + 0.2 * loss_res1
    loss = loss_cons_total + loss_res_total

    return loss


def loss_func1(noisy_img , model, method_type):

    noisy1, noisy2 = pair_downsampler(noisy_img)
    noisy3, noisy4 = pair_downsampler5(noisy_img)
    noisy5, noisy6 = pair_downsampler1(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    pred3 = noisy3 - model(noisy3)
    pred4 = noisy4 - model(noisy4)
    pred5 = noisy5 - model(noisy5)
    pred6 = noisy6 - model(noisy6)

    loss_res = 1 / 6 * (mse(noisy1, pred2) + mse(noisy2, pred1) + mse(noisy3,pred4) + mse(noisy4,pred3) + mse(noisy5,pred6) + mse(noisy6, pred5))

    noisy_denoised = noisy_img - model(noisy_img)
    if method_type == 'modified':
        denoised1, denoised2 = pair_downsampler(noisy_denoised)
        denoised3, denoised4 = pair_downsampler5(noisy_denoised)
        denoised5, denoised6 = pair_downsampler1(noisy_denoised)
    else:
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 6 * (mse(pred1, denoised1) + mse(pred2, denoised2) + (mse(pred3,denoised3)
    + (mse(pred4, denoised4)) + mse(pred5, denoised5) + mse(pred6, denoised6)))

    #new_loss

    loss_cons1 = 1 / 6 * (mse(noisy1, denoised1) + mse(noisy2, denoised2) + (mse(noisy3,denoised3)
    + (mse(noisy4, denoised4)) + mse(noisy5, denoised5) + mse(noisy6, denoised6)))


    loss = loss_res + loss_cons
    return loss

def loss_func2(noisy_img , model, method_type):

    noisy1, noisy2 = pair_downsampler1(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    if method_type == 'modified':
        denoised1, denoised2 = pair_downsampler1(noisy_denoised)
    else:
        denoised1, denoised2 = pair_downsampler1(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss

def loss_func33(noisy_img , model, method_type, down1=None , down2=None):

    noisy1, noisy2 = pair_downsampler1(noisy_img)


    if down1 is not None and down2 is not None:
        mean_sample = torch.mean(torch.stack([down1, down2]), dim=0)
        noisy1,noisy2 = pair_downsampler3(mean_sample)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    if method_type == 'modified':
        denoised1, denoised2 = pair_downsampler1(noisy_denoised)
    else:
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss

def loss_func3(noisy_img , model, method_type, down1=None , down2=None):

    noisy1, noisy2 = pair_downsampler(down1)
    noisy11, noisy22 = pair_downsampler(down2)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    pred11 = noisy11 - model(noisy11)
    pred22 = noisy22 - model(noisy22)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))
    loss_res1 = 1 / 2 * (mse(noisy11, pred22) + mse(noisy22, pred11))
    noisy_denoised = noisy_img - model(noisy_img)

    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons + 0.2*loss_res1

    return loss

def loss_residual (model , noisy_image1, noisy_image2):
    pred1 = noisy_image1 - model(noisy_image1)
    pred2 = noisy_image2 - model(noisy_image2)
    loss_res = 1 / 2 * (mse(noisy_image1, pred2) + mse(noisy_image2, pred1))
    return pred1,pred2,loss_res

def loss_consistency(model, noisy_img,pred1, pred2, denoised1, denoised2):
    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))
    return loss_cons