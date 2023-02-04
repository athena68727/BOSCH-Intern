from cgi import test
from pickletools import uint8
from random import sample
from webbrowser import get
from xmlrpc.client import Boolean
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from seg_dataset import segdataset
from unet import get_model
#from unetpp import get_modelpp, dice_loss
import cv2
import datetime


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def dice_loss(output, batch_label, smooth = 1.):
    output = output.contiguous()
    batch_label = batch_label.contiguous()

    intersection = (output * batch_label).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (output.sum(dim=2).sum(dim=2) + batch_label.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()



def train(args):
    data = segdataset(r"dataset_154/")
    train_size = int(data.len*args.split)
    test_size = data.len - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle=False)

    if args.model == "unet":
        model = get_model(4, True)
    else:
        model = get_modelpp(4, True)
    print(model)
    weights = torch.tensor(np.load(r'dump/classes_weights.npy')).to(device)
    loss_func = torch.nn.BCELoss(weight = weights)
    optim = torch.optim.SGD(model.parameters(), lr = args.lr)
    lr_scheduler = StepLR(optim, step_size=args.lr_step_size, gamma=args.gamma)

    best_loss = 10000
    best_lst = [10000, 10000, 10000, 10000]
    now = datetime.datetime.now()
    dir = r"log/exp" + now.strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(dir)

    for i in range(args.epoch):
        start = time.time()
        model.train()

        total_loss = 0
        loss_lst = [0, 0, 0, 0]
        with tqdm(range(len(train_loader)), desc='Train epoch '+str(i)) as tbar:
            for step, (batch_image, batch_label) in enumerate(train_loader):
                batch_image = batch_image.to(device)
                batch_label = batch_label.to(device)
                #print(batch_image[0].transpose(0,1).transpose(1,2).shape) #c,w,h -> w,c,h -> w,h,c
                outputs = model(batch_image)
                if args.model == "unet":
                    outputs = outputs.to(torch.float32)
                    batch_label = batch_label.to(torch.float32)

                    optim.zero_grad()
                    outputs = outputs.transpose(1,2).transpose(2,3)
                    batch_label = batch_label.transpose(1,2).transpose(2,3)
                    #print(outputs.shape)
                    loss = loss_func(outputs, batch_label)
                    if args.use_dice:
                        loss += dice_loss(outputs, batch_label)

                    loss.backward()
                    optim.step()
                    lr_scheduler.step()
                    tbar.update()
                    total_loss += loss.item()

                else:
                    loss_sum = torch.tensor(0).to(torch.float32).to(device)
                    batch_label = batch_label.to(torch.float32)
                    batch_label = batch_label.transpose(1,2).transpose(2,3)

                    for n, output in enumerate(outputs):
                        output = output.to(torch.float32)
                        output = output.transpose(1,2).transpose(2,3)

                        loss = loss_func(output, batch_label)

                        if args.use_dice:
                            loss += dice_loss(output, batch_label)

                        loss_lst[n] += loss.item()
                        loss_sum += loss

                    optim.zero_grad()
                    loss_sum.backward()
                    optim.step()
                    lr_scheduler.step()
                    tbar.update()


        model.eval()
        test_total_loss = 0
        right_number = 0
        result = []
        with torch.no_grad():
            for test_data  in test_loader:
                test_i, test_l = test_data
                test_i = test_i.to(device)
                test_l = test_l.to(device)
                test_o = model(test_i)

                if args.model == "unet":
                    test_o = test_o.to(torch.float32)
                    #test_o = torch.nn.functional.log_softmax(test_o, dim = 1)
                    test_l = test_l.to(torch.float32)
                    #test_l = test_l.long()

                    #right_number += torch.eq(torch.argmax(test_o, dim = 1), torch.argmax(test_l, dim = 1)).sum()
                    #right_number += torch.eq(torch.argmax(test_o, dim = 1), test_l).sum()
                    sample_img = test_i[0].transpose(0,1).transpose(1,2).cpu().numpy()*255
                    sample_img = sample_img.astype(np.uint8)
                    out = test_o[0].transpose(0,1).transpose(1,2).cpu().numpy()
                    test_l = test_l[0].transpose(0,1).transpose(1,2).cpu().numpy()
                    labels = np.argmax(out, axis=2)
                    gt_labels = np.argmax(test_l, axis = 2)
                    acc = np.sum(labels == gt_labels)/(labels.shape[0]*labels.shape[1])

                    mask = np.zeros_like(sample_img)
                    mask = mask.astype(np.uint8)

                    # ugly implement, can -> blue, button -> green, line -> red
                    for p in range(mask.shape[0]):
                        for q in range(mask.shape[1]):
                            if labels[p][q] == 1:
                                mask[p][q][0] = 255
                            elif labels[p][q] == 2:
                                mask[p][q][1] = 255
                            elif labels[p][q] == 3:
                                mask[p][q][2] = 255

                    result.append(cv2.addWeighted(src1=sample_img, \
                        alpha=0.5, src2=mask, beta=0.5, gamma=0))
                else:

                    test_l = test_l.to(torch.float32)
                    test_l = test_l[0].transpose(0,1).transpose(1,2).cpu().numpy()
                    gt_labels = np.argmax(test_l, axis = 2)
                    for n in range(4):
                        o = test_o[n]
                        o = o[0]
                        o = o.to(torch.float32)
                        sample_img = test_i[0].transpose(0,1).transpose(1,2).cpu().numpy()*255
                        sample_img = sample_img.astype(np.uint8)
                        out = o.transpose(0,1).transpose(1,2).cpu().numpy()
                        labels = np.argmax(out, axis=2)
                        acc = np.sum(labels == gt_labels)/(labels.shape[0]*labels.shape[1])

                        mask = np.zeros_like(sample_img)
                        mask = mask.astype(np.uint8)

                        # ugly implement, can -> blue, button -> green, line -> red
                        for p in range(mask.shape[0]):
                            for q in range(mask.shape[1]):
                                if labels[p][q] == 1:
                                    mask[p][q][0] = 255
                                elif labels[p][q] == 2:
                                    mask[p][q][1] = 255
                                elif labels[p][q] == 3:
                                    mask[p][q][2] = 255

                        result.append(cv2.addWeighted(src1=sample_img, \
                            alpha=0.5, src2=mask, beta=0.5, gamma=0))




        end = time.time()
        if args.model == "unet":
            print("Epoch:{}, traning total loss:{}, test accuracy:{}".format(i, total_loss, acc))
            print("Time cost for this epoch:{}, estimated time left:{}".format(end - start, (end - start)*(args.epoch - i - 1)))

            if total_loss < best_loss:
                torch.save(model, os.getcwd() + r'/ckpt/unet.pth')
                best_loss = total_loss
                cv2.imwrite(dir + r"/" + str(i) + ".jpg", result[0])
                print("Best model saved!")

        else:
            print("Epoch:{}, traning total loss:{}, test accuracy:{}".format(i, loss_lst, acc))
            print("Time cost for this epoch:{}, estimated time left:{}".format(end - start, (end - start)*(args.epoch - i - 1)))

            for n, l in enumerate(loss_lst):
                if l < best_lst[n]:
                    torch.save(model, os.getcwd() + '/ckpt/unetpp_L{}.pth'.format(n+1))
                    best_lst[n] = l
                    print("Best L{} model saved!".format(n+1))
                    cv2.imwrite(dir + r"/" + str(i) + "L{}.jpg".format(n+1), result[n])


    print("Model saved!")



# For training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--lr_step_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--split", type=float, default=0.99)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--use_dice", type=Boolean, default="False")
    args = parser.parse_args()
    train(args)



