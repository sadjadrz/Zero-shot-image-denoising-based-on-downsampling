import torch
import torch.nn.functional as F
import torch.nn as nn
from Utils.img_utils import filter_Tensor


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)

    filter1 = filter1.repeat(c, 1, 1, 1)


    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)



    return output1, output2

def pair_downsampler1(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, .35], [0.65, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[.35, 0], [0, 0.65]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def pair_downsampler2(img):
    # img has shape B C H W
    c = img.shape[1]
    #img2 = filter_Tensor(img , 'SHARPEN')
    #img1 = filter_Tensor(img , 'SMOOTH')
    filter1 = torch.FloatTensor([[[[0, 0.35 , 0], [0.65, 0 ,0]]]]).to(img.device)
    #filter1 = torch.clamp(filter1)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter11 = torch.FloatTensor([[[[0, 0.65], [0.35, 0]]]]).to(img.device)
    filter11 = filter11.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, dilation=1, groups=c)
    output11 = F.conv2d(img, filter11, stride=2, dilation=1, groups=c)
    output1 = F.pad(output1, (0, 1, 0, 0))
    #output1 = filter_Tensor(output1, 'BLUR', False)



    filter2 = torch.FloatTensor([[[[0.5, 0 , 0], [0,0.5 ,0]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    filter22 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter22 = filter22.repeat(c, 1, 1, 1)

    output2 = F.conv2d(img, filter2, stride=2, dilation=1, groups=c)
    output22 = F.conv2d(img, filter22, stride=2, dilation=1, groups=c)
    output2 = F.pad(output2, (0, 1, 0, 0))
    #output2 = filter_Tensor(output2, 'SHARPEN', False)
    output2 = torch.mean(torch.stack([output2, output22]), dim=0)
    output1 = torch.mean(torch.stack([output1, output11]), dim=0)
    return output1, output2

def pair_downsampler3(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)

    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2,dilation=1, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, dilation=1, groups=c)
    return output1,output2

def pair_downsampler4(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5 , 0], [0.5, 0 ,0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0 , 0], [0, 0.5 ,0]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, dilation=1, groups=c)
    #output1 = F.pad(output1, (0, 1, 0, 0))
    #output1 = filter_Tensor(output1, 'MinFilter', False)

    output2 = F.conv2d(img, filter2, stride=2, dilation=1, groups=c)
    #output2 = F.pad(output2, (0, 1, 0, 0))
    #output2 = filter_Tensor(output2, 'SHARPEN', False)

    return output1, output2

def pair_downsampler5(img):
        # img has shape B C H W
        c = img.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.45], [0.55, 0]]]]).to(img.device)


        filter1 = filter1.repeat(c, 1, 1, 1)


        filter2 = torch.FloatTensor([[[[0.55, 0], [0, 0.45]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)
        return output1, output2

def pair_downsampler6(img):
    # img has shape B C H W
    c = img.shape[1]
    #img2 = filter_Tensor(img , 'SHARPEN')
    #img1 = filter_Tensor(img , 'SMOOTH')
    filter1 = torch.FloatTensor([[[[0, 0.5 ], [0, 0]]]]).to(img.device)
    #filter1 = torch.clamp(filter1)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter11 = torch.FloatTensor([[[[0, 0], [0.5, 0]]]]).to(img.device)
    filter11 = filter11.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, dilation=1, groups=c)
    output11 = F.conv2d(img, filter11, stride=2, dilation=1, groups=c)
    output1 = F.pad(output1, (0, 1, 0, 0))
    #output1 = filter_Tensor(output1, 'BLUR', False)



    filter2 = torch.FloatTensor([[[[0.5, 0 ], [0 ,0]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    filter22 = torch.FloatTensor([[[[0, 0], [0, 0.5]]]]).to(img.device)
    filter22 = filter22.repeat(c, 1, 1, 1)

    output2 = F.conv2d(img, filter2, stride=2, dilation=1, groups=c)
    output22 = F.conv2d(img, filter22, stride=2, dilation=1, groups=c)
    output2 = F.pad(output2, (0, 1, 0, 0))
    #output2 = filter_Tensor(output2, 'SHARPEN', False)
    output2 = torch.mean(torch.stack([output2, output22]), dim=0)
    output1 = torch.mean(torch.stack([output1, output11]), dim=0)
    return output1, output2

def pair_downsampler7(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, .5], [0,.0]]]]).to(img.device)
    filter11 = torch.FloatTensor([[[[0, 0], [.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter11 = filter11.repeat(c, 1, 1, 1)
    filter1 = torch.mean(torch.stack([filter1, filter11]), dim=0)
    filter2 = torch.FloatTensor([[[[.5, 0], [0,0]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

def pair_downsampler8(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, -0.44], [0.55, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter1 = torch.clamp(filter1,min=0)

    filter2 = torch.FloatTensor([[[[0.55, 0], [0, -0.44]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    filter2 = torch.clamp(filter2,min=0)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def pair_downsampler9(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, .45], [0,.55]]]]).to(img.device)
    filter11 = torch.FloatTensor([[[[0, 0.55], [0.45, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter11 = filter11.repeat(c, 1, 1, 1)
    filter1 = torch.mean(torch.stack([filter1, filter11]), dim=0)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0,0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

def pair_downsampler10(img):
    # img has shape B C H W
    c = img.shape[1]
    #img2 = filter_Tensor(img , 'SHARPEN')
    #img1 = filter_Tensor(img , 'SMOOTH')
    filter1 = torch.FloatTensor([[[[0, 0.45 ], [0.55, 0]]]]).to(img.device)
    #filter1 = torch.clamp(filter1)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter11 = torch.FloatTensor([[[[0, 0.55], [0.45, 0]]]]).to(img.device)
    filter11 = filter11.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, dilation=1, groups=c)
    output11 = F.conv2d(img, filter11, stride=2, dilation=1, groups=c)
    output1 = F.pad(output1, (0, 1, 0, 0))
    #output1 = filter_Tensor(output1, 'BLUR', False)

    filter2 = torch.FloatTensor([[[[0.45, 0 ], [0 ,0.55]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    filter22 = torch.FloatTensor([[[[0.55, 0], [0, 0.45]]]]).to(img.device)
    filter22 = filter22.repeat(c, 1, 1, 1)

    output2 = F.conv2d(img, filter2, stride=2, dilation=1, groups=c)
    output22 = F.conv2d(img, filter22, stride=2, dilation=1, groups=c)
    output2 = F.pad(output2, (0, 1, 0, 0))
    #output2 = filter_Tensor(output2, 'SHARPEN', False)
    output2 = torch.mean(torch.stack([output2, output22]), dim=0)
    output1 = torch.mean(torch.stack([output1, output11]), dim=0)
    return output1, output2

def pair_downsampler11(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[.1, -0.44], [0.55, 0.1]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter1 = torch.clamp(filter1,min=0)

    filter2 = torch.FloatTensor([[[[0.55, .1], [.1, -0.44]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    filter2 = torch.clamp(filter2,min=0)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


