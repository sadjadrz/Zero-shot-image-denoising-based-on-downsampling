import torch
import os
import  numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN, GaussianBlur,MinFilter
)

def add_noise(x, noise_level, noise_type):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level

    return noisy

def imgFilePath_To_Tensor(img_path, unsqueeze ):
    resize_transform = transforms.CenterCrop((256, 256))
    img = Image.open(img_path)
    resized_img = resize_transform(img)
    transform = transforms.ToTensor()
    tensor_image = transform(resized_img)
    if unsqueeze:
        tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

def ToTensor(img , resize):
    if resize:
        resize_transform = transforms.CenterCrop((128, 128))
        img = resize_transform(img)
    transform = transforms.ToTensor()
    tensor_image = transform(img)
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

def filter_Tensor(tensor_img , filter_type , resize):
    img = tensor_img.squeeze(0)
    img_pil = to_pil_image(img)
    if filter_type == 'BLUR':
        img = img_pil.filter(BLUR)
    if filter_type == 'EDGE_ENHANCE':
        img = img_pil.filter(EDGE_ENHANCE)
    if filter_type == 'SMOOTH':
        img = img_pil.filter(SMOOTH)
    if filter_type == 'SHARPEN':
        img = img_pil.filter(SHARPEN)
    if filter_type == 'GaussianBlur':
        img = img_pil.filter(GaussianBlur)
    if filter_type == 'MinFilter':
        img = img_pil.filter(MinFilter)
    img = ToTensor(img, resize).to('cuda')
    return img

def convert_ext_png(directory):
    for filename in os.listdir(directory):
        if not filename.endswith(".png") :
            try:

                # Open the image file
                img = Image.open(os.path.join(directory, filename))

                # Convert the image to PNG format (if not already in PNG)
                if img.format != 'PNG':
                    new_filename = os.path.splitext(filename)[0] + '.png'
                    img.save(os.path.join(directory, new_filename))

                # Close the image file
                img.close()
            except Exception as e:
                print(f"An error occurred while processing {filename}: {str(e)}")
