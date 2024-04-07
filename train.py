import torch.optim as optim
from model.Network import network, network1
from Utils.options import parse
from Utils.img_utils import imgFilePath_To_Tensor, add_noise
from Utils.img_downsampler import pair_downsampler , pair_downsampler1
from metrics.loss import loss_func, mse
import numpy as np
import argparse
import os
from PIL import  Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt)
    return opt


def show_images(noisy_img, downsampler1, downsampler2):
    img0 = noisy_img.cpu().squeeze(0).permute(1, 2, 0)
    img1 = downsampler1.cpu().squeeze(0).permute(1, 2, 0)
    img2 = downsampler2.cpu().squeeze(0).permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    ax[0].imshow(img0)
    ax[0].set_title('Noisy Img')

    ax[1].imshow(img1)
    ax[1].set_title('First downsampled')

    ax[2].imshow(img2)
    ax[2].set_title('Second downsampled')
    plt.show()

def save_image( img_tensor, filename):
    img0 = img_tensor.cpu().squeeze(0).permute(1, 2, 0)
    image_array = img0.cpu().numpy()
    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_array)
    file_path = os.path.join(opt['output_dir'], filename)
    image_pil.save(file_path)


def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img, model)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test1(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)

    return PSNR

def test(noisy_img, clean_img):
    with torch.no_grad():
        MSE = mse(clean_img, noisy_img).item()
        PSNR = 10 * np.log10(1 / MSE)

    return PSNR

def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred
if __name__ == '__main__':
    opt = parse_options(is_train=True)
    data_dir = opt['data_dir']
    img_dir = opt['img_dir']
    img_name = opt['img_name']
    img_ext = opt['img_ext']
    noise_level = opt['noise_level']
    noise_type = opt['noise_type']
    img_path = data_dir + img_dir + img_name + img_ext
    clean_img = imgFilePath_To_Tensor(img_path, True)

    noisy_img = add_noise(clean_img, noise_level, noise_type)

    clean_img = clean_img.to(opt['device'])
    noisy_img = noisy_img.to(opt['device'])

    img1, img2 = pair_downsampler(noisy_img)

    #show_images(noisy_img, img1, img2)
    save_image(noisy_img,"noisy.png")
    save_image(img1,"down1.png")
    save_image(img2,"down2.png")
    save_image(clean_img,"clean.png")

    model = network1(clean_img.shape[1])
    model.to(opt['device'])
    print("The number of parameters of the network is: ",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    max_epoch = opt['max_epoch']  # training epochs
    lr = opt['lr']  # learning rate
    step_size = opt['step_size']  # number of epochs at which learning rate decays
    gamma = opt['gamma']  # factor by which learning rate decays

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(max_epoch)):
        train(model, optimizer, noisy_img)
        scheduler.step()

    PSNR_predict = test1(model, noisy_img, clean_img)
    PSNR_nois = test(noisy_img, clean_img)
    print('predict__clean__' + img_name + '__' + str(PSNR_predict))
    print('noisy__clean__' + img_name + '__' + str(PSNR_nois))

    denoised_img = denoise(model, noisy_img)
    save_image(denoised_img, "denoised.png")