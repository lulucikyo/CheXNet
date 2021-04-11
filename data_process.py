import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time; _START_RUNTIME = time.time()
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class XrayDataSet(Dataset):
    def __init__(self, data_path, image_list):
        self.image_path = []
        self.y=[]
        f = open(image_list, "r")
        for line in f:
            l = line.strip("\n").split(" ")
            self.image_path.append(data_path+l[0])
            label = [int(x) for x in l[1:]]
            self.y.append(label)
        f.close()
    def __len__(self):
        return(len(self.image_path))
    def __getitem__(self, index):
        return(self.image_path[index], self.y[index])


def collate_fn(data):
    image_path, label = zip(*data)
    image_tensors = torch.Tensor()
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
                ])
    for img in image_path:
        img_pil = Image.open(img).convert("RGB")
        #print(img_pil)
        img_tensor = trans(img_pil).unsqueeze(0)
        #print(img_tensor)
        image_tensors = torch.cat((image_tensors, img_tensor))


    label_tensors = torch.FloatTensor(label)

    return image_tensors, label_tensors