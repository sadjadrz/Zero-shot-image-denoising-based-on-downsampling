from model.ResNet import ResnetBlock
import torch.nn as nn
import torchvision.models as models
class network1(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network1, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        #print(f'{x.shape}first')
        x = self.act(self.conv2(x))
        #print(f'{x.shape}second')
        #x = self.act(self.conv2(x))
        x = self.conv3(x)
        #print(f'{x.shape}third')

        return x


class network(nn.Module):
    def __init__(self, n_chan, chan_embed=24):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.twocon1 = nn.Conv2d(n_chan, chan_embed, kernel_size=3, padding=1)
        self.resnet_block1 = ResnetBlock(n_chan, chan_embed)

        self.twocon2 = nn.Conv2d(chan_embed, chan_embed, kernel_size=3, padding=1)
        self.resnet_block2 = ResnetBlock(chan_embed, chan_embed)

        self.twocon3 = nn.Conv2d(chan_embed, n_chan, kernel_size=1, padding=0)
        self.resnet_block3 = ResnetBlock(chan_embed, n_chan, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = x
        # print(x.shape)
        x = self.act(self.twocon1(x))
        # print(x.shape)
        x2 = self.resnet_block1(x1)
        # print(x2.shape)
        x = x2 + x

        return x


