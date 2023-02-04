from itertools import count
import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def calculate_weigths_labels(dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    print('Calculating classes weights')
    tqdm_batch = tqdm(dataloader)
    for _, y in tqdm_batch:
        y = y.detach().cpu().numpy()[0]
        count_l = np.sum(y, axis = 2)
        count_l = np.sum(count_l, axis = 1)
        #print(count_l)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    print(ret)
    classes_weights_path = r'dump/classes_weights.npy'
    np.save(classes_weights_path, ret)

    return ret

class segdataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.img = os.listdir(os.path.join(path, "img"))
        #self.mask = os.listdir(os.path.join(path, "mask"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.len = len(self.img)

    def __len__(self):
        return self.len

    def __trans__(self, img, size):
        # transform to square and padding
        h, w = img.shape[0:2]
        _w = _h = size
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def process_mask(self):
        for index in range(self.len):
            img = self.img[index]
            img_path = [os.path.join(self.path, i) for i in ("img", "mask")]
            #img_o = cv2.imread(os.path.join(img_path[0], img))
            #_, img_l = cv2.VideoCapture(os.path.join(img_path[1], mask)).read()
            img_l = cv2.imread(os.path.join(img_path[1], img[:-4]+"_m.jpg"))
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            img_l = self.__trans__(img_l, 1440)

            label = np.zeros((img_l.shape[0], img_l.shape[1], 4))
            
            for w  in range(img_l.shape[0]):
                for h in range(img_l.shape[1]):
                    if img_l[w][h] == 0:
                        label[w][h][0] = 1
                    if img_l[w][h] == 50:
                        label[w][h][1] = 1
                    if img_l[w][h] == 100:
                        label[w][h][2] = 1
                    if img_l[w][h] == 200:
                        label[w][h][3] = 1

            np.save(os.path.join(img_path[1], img[:-4]+"_m.npy"), label)

        # 转成网络需要的正方形
        #img_o = self.__trans__(img_o, 1440)
        #label = self.__trans__(label, 1440)
        return       

    def __getitem__(self, index):
        # 拿到的图片和标签
        img = self.img[index]
        img_path = [os.path.join(self.path, i) for i in ("img", "mask")]
        img_o = cv2.imread(os.path.join(img_path[0], img))
        #_, img_l = cv2.VideoCapture(os.path.join(img_path[1], mask)).read()
        img_l = np.load(os.path.join(img_path[1], img[:-4] + "_m.npy"))

        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, 1440)
        #label = self.__trans__(label, 1440)

        return self.trans(img_o), self.trans(img_l)

if __name__ == "__main__":
    data = segdataset(r"dataset_154/")
    dataloader = DataLoader(dataset = data, batch_size = 1, shuffle=False)
    data.process_mask()
    calculate_weigths_labels(dataloader=dataloader, num_classes=4)
