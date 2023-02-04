from cgi import test
from webbrowser import get
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from dataset import detectdataset
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torch import nn
from collections import OrderedDict

'''
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}'''



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_model(model_pth, for_training = True):
    if for_training:
        model = models.mobilenet_v2()
        pthfile = os.getcwd() + r'/ckpt/mobilenet_v2-b0353104.pth'
        model.load_state_dict(torch.load(pthfile))
        model.classifier[-1] = torch.nn.Linear(1280,2)
        device_ids = [0]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model.to(device)
    else:
        '''model = models.resnet18(pretrained=False)
        
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )'''
        model = models.mobilenet_v2()
        model.classifier[-1] = torch.nn.Linear(1280,2)

        pthfile = model_pth
        state_dict = torch.load(pthfile, map_location=torch.device('cpu'))
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            #print(name)
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

        '''model = nn.DataParallel(model, device_ids = ['cpu'])
        pthfile = os.getcwd() + model_pth
        #model = torch.load(pthfile, map_location=torch.device('cpu'))
        model_dict = torch.load(pthfile, map_location=torch.device('cpu')).module.state_dict()
        model.module.load_state_dict(model_dict)'''
        return model
    #print(model.classifier)

def train(args):
    data = detectdataset(args.part + r"/list.txt", args.part + r'/')
    train_size = int(data.len*args.split)
    test_size = data.len - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle=False) 
    model = get_model(None, True)
    print(model)
    optim = torch.optim.SGD(model.parameters(), lr = args.lr)
    weight = torch.tensor([1.0, args.weight]).to(device).to(torch.float32)
    loss_f = torch.nn.CrossEntropyLoss(weight = weight)
    lr_scheduler = StepLR(optim, step_size=args.lr_step_size, gamma=args.gamma)

    best_acc = None
    best_loss = None
    best_alarm = 1

    for i in range(args.epoch):
        start = time.time()
        model.train()
        with tqdm(range(len(train_loader)), desc='Train epoch '+str(i)) as tbar:
            for step, (batch_image, batch_label) in enumerate(train_loader):
                batch_image = batch_image.to(device)
                batch_label = batch_label.to(device)
                outputs = model(batch_image)
                outputs = outputs.to(torch.float32)
                #outputs = torch.nn.functional.log_softmax(outputs, dim = 1)
                #print(torch.argmax(outputs, dim = 1))
                batch_label = batch_label.to(torch.float32)
                batch_label = batch_label.long()
                #print(batch_label)

                optim.zero_grad()

                loss = loss_f(outputs, batch_label)
                #print(loss.item())
                #loss.requires_grad_(True)
                loss.backward()
                optim.step()
                if loss.item() > 0.3 or step%3 == 0:
                    lr_scheduler.step()
                tbar.update()
                #break

        model.eval()
        test_total_loss = 0
        right_number = 0
        false_neg = 0
        total_ok = 0
        total_neg = 0
        fail_alarm = 0

        with torch.no_grad():
            for test_data  in test_loader:
                test_i, test_l = test_data
                test_i = test_i.to(device)
                test_l = test_l.to(device)
                test_o = model(test_i)
                test_o = test_o.to(torch.float32)
                #test_o = torch.nn.functional.log_softmax(test_o, dim = 1)
                test_l = test_l.to(torch.float32)
                test_l = test_l.long()
                test_total_loss += loss_f(test_o, test_l).item()

                #right_number += torch.eq(torch.argmax(test_o, dim = 1), torch.argmax(test_l, dim = 1)).sum()
                o = torch.argmax(test_o, dim = 1)
                right_number += torch.eq(o, test_l).sum()
                #print(o.shape)
                #print(test_l.shape)
                total_ok += (test_l == 1).sum()
                total_neg += (test_l == 0).sum()
                false_neg += (o[test_l == 1] == 0).sum()
                fail_alarm += (o[test_l == 0] == 1).sum()

        end = time.time()
        print("Epoch:{}, loss:{}, accuracy:{}, false alarm rate:{}, fail alarm rate:{}".format(i, test_total_loss, (right_number/test_size), false_neg/total_ok, fail_alarm/total_neg))
        print("Time cost for this epoch:{}, estimated time left:{}".format(end - start, (end - start)*(args.epoch - i - 1)))
        
        '''best_acc is None or best_acc < right_number/test_size or (best_acc == right_number/test_size\
            and test_total_loss < best_loss) or '''
        if best_loss is None or test_total_loss < best_loss:
            torch.save(model.state_dict(), os.getcwd() + r'/ckpt/'+ args.part + '_mobilenet_v2-grease-detection.pth')
            best_acc = right_number/test_size
            best_loss = test_total_loss
            best_alarm = false_neg/total_ok
            print("Best model saved!")
    
    print("Model saved!")



# For training, remember to generate list
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument("--part", type=str, default="can")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--lr_step_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--split", type=float, default=0.9)
    args = parser.parse_args()
    print(device)    
    train(args)