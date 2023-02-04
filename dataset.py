import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class detectdataset(Dataset):
    def __init__(self, list, image_dir):
        self.image_label_list = []
        with open(list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                if int(content[1]):
                    labels= 1#[0, 1] # For lower torch then 1
                else:
                    labels = 0#[1, 0] # For lower torch then 0
                self.image_label_list.append((name, labels))
        random.shuffle(self.image_label_list)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
    
    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        img = Image.open(self.image_dir + image_name)
        img = preprocess(img)
        label=np.array(label)
        return img, label

    def __len__(self):
        return self.len