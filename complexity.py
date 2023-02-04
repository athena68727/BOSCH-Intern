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
#from unetpp import getL3, get_modelpp
from unet import get_model
from thop import profile


origin_unet = get_model(training=False)
L3net = getL3()
L4net = get_modelpp(training = False)
input = torch.randn(1, 3, 1440, 1440)

flops, params = profile(origin_unet, (input,))
print('Origin_unet flops: ', flops, 'params: ', params)

flops, params = profile(L4net, (input,))
print('L4_unet flops: ', flops, 'params: ', params)

flops, params = profile(L3net, (input,))
print('L3_unet flops: ', flops, 'params: ', params)