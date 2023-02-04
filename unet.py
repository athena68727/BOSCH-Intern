from cgi import test
from webbrowser import get
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False), # kernel size = 1 -> MLP
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out



# This U-Net contains attention module
class UnetResNet(nn.Module):
    def __init__(self, n_class, out_size):
        super().__init__()
 
        self.out_size = out_size
        self.backbone = models.resnet18(pretrained=False)
        self.baseLayer = list(self.backbone.children())
 
        self.layer0 = nn.Sequential(*self.baseLayer[:3])
        self.layer1 = nn.Sequential(*self.baseLayer[3:5])
        self.layer2 = self.baseLayer[5]
        self.layer3 = self.baseLayer[6]
        self.layer4 = self.baseLayer[7]

        self.cbam0 = CBAM(channel=64)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)

        self.layerup3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.layerup2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.layerup1 = nn.ConvTranspose2d(128, 64, 2, 2 )
        self.layerup0 = nn.ConvTranspose2d(64, 64, 2, 2 )
 
        self.layerupConV3 = ConvBlock(512, 256)#nn.Conv2d(512, 256, 3)
        self.layerupConV2 = ConvBlock(256, 128)#nn.Conv2d(256, 128, 3)
        self.layerupConV1 = ConvBlock(128, 64)#nn.Conv2d(128, 64, 3)
        self.layerupConV0 = ConvBlock(128, 64)#nn.Conv2d(128, 64, 3)
 
        self.ReLu = nn.ReLU()
 
        self.endlayer = nn.Conv2d(64, n_class, 1)
 
    def Crop(self, encode_features, x):
        _, _, H, W = x.shape
        return torchvision.transforms.CenterCrop([H, W])(encode_features)
 
 
    def forward(self, input_x):
 
        layer0 = self.layer0(input_x)
        layer0 = self.cbam0(layer0) + layer0

        layer1 = self.layer1(layer0)
        layer1 = self.cbam1(layer1) + layer1

        layer2 = self.layer2(layer1)
        layer2 = self.cbam2(layer2) + layer2

        layer3 = self.layer3(layer2)
        layer3 = self.cbam3(layer3) + layer3

        layer4 = self.layer4(layer3)
 
        x = self.layerup3(layer4)
        #print(x.shape)
        #print(layer3.shape)
        layer3skip = self.Crop(layer3, x)
        x = torch.cat([x, layer3skip], dim=1)
        x = self.layerupConV3(x)
 
        x = self.layerup2(x)
        layer2skip = self.Crop(layer2, x)
        x = torch.cat([x, layer2skip], dim=1)
        x = self.layerupConV2(x)
 
        x = self.layerup1(x)
        layer1skip = self.Crop(layer1, x)
        x = torch.cat([x, layer1skip], dim=1)
        x = self.layerupConV1(x)

        x = self.layerup0(x)
        layer0skip = self.Crop(layer0, x)
        x = torch.cat([x, layer0skip], dim=1)
        x = self.layerupConV0(x)

        x = self.endlayer(x)
        x = torch.sigmoid(x)
 
        out = F.interpolate(x, self.out_size)
 
        return out

def get_model(num_class: object = 4, training: object = True) -> object:
    if training:
        state_dict_path = r"ckpt/resnet18-5c106cde.pth"
        model = UnetResNet(num_class, [1440, 1440]).to(device)
        #model = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=pretrained)
        state_dict = torch.load(state_dict_path)
        new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                    if not (k.startswith('l')
                            | k.startswith('fc'))}  # # discard MLP and fc
        model_dict = model.state_dict()
        
        model_dict.update(new_dict)
        model.load_state_dict(model_dict, strict=False)
        model = nn.DataParallel(model, device_ids=device_ids)
        #total_params = sum(p.numel() for p in model.parameters())
        #print(f'{total_params:,} total parameters.')

        return model.to(device)

    else:
        state_dict_path = r"ckpt/unet.pth"
        model = UnetResNet(num_class, [1440, 1440]).to(device)
        #model = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=pretrained)
        model_dict = torch.load(state_dict_path, map_location=torch.device('cpu')).module.state_dict()
        model = nn.DataParallel(model)
        model.module.load_state_dict(model_dict)
        total_params = sum(p.numel() for p in model.parameters())
        #print(f'{total_params:,} total parameters.')

        return model.to(device)