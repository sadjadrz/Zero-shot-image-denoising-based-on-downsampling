import torch
import  pywt
import torch.nn.functional as F
from Utils.img_downsampler import *

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

 
