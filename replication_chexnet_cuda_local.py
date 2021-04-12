# -*- coding: utf-8 -*-
"""Replication CheXNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-S1WhqiqsChuWVGjunT_5CIGdlV-kHO3
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""You want to use the full path for Colab"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/DL4H Project/replication/images_converted/
#!(ls | wc -l)
# %cd /content/drive/My Drive/DL4H Project/replication

"""Yes, there is some dups in the dataset, but it does not affect"""

import os
import gc
import random
import time; _START_RUNTIME = time.time()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import zipfile

# # extract_path = "./image_converted/"
# filename = "images_converted.zip"
# print(filename)
# zip = zipfile.ZipFile(filename)
# zip.extractall(".")

# CUDA stats
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

N_CLASSES = 14
CLASS_NAMES = ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", 
               "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
               "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
DATA_DIR = './images_converted/'
#DATA_DIR = '/content/drive/My Drive/DL4H Project/replication/images_converted/'
TEST_IMAGE_LIST = 'labeled_test_list.txt'
BATCH_SIZE = 16
PRINT_INTERVAL = 50

"""BATCH_SIZE -> 8 is way better than 16 in Colab"""

# img_list_ = os.listdir("./images_converted/")
# print(len(img_list_))

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

    return image_tensors.cuda(), label_tensors.cuda()

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

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def train_model(model, train_loader, n_epochs = 1):
    t1 = time.time()
    criterion = nn.BCELoss()
    """using Adam with standard parameters (B1 = 0:9 and B2 = 0:999) """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    """factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1."""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                               patience=10, verbose=False, threshold=0.0001,
                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    # prep model for training
    model.train()

    train_loss_arr = []
    print("Started training, total epoch : {}".format(n_epochs))
    print("Training data size: {}".format(len(train_loader)))
    for epoch in range(n_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = 0
        batch = 0
        print("Started epoch {}".format(epoch))
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            #print(y.shape)
            #print(y_hat.shape)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch % PRINT_INTERVAL == 0):
                print('Trained {} batches \tTraining Loss: {:.6f}'.format(batch, loss.item()))
            batch += 1

        train_loss = train_loss / len(train_loader)
        scheduler.step(train_loss)
        if epoch % 1 == 0:
            train_loss_arr.append(np.mean(train_loss))
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    t2 = time.time()
    print("Training time lapse: {} min".format((t2 - t1) // 60))

def eval_model(model, test_loader):
    # initialize the y_test and y_pred tensor
    y_test = torch.FloatTensor()
    y_test = y_test.cuda()
    y_pred = torch.FloatTensor()
    y_pred = y_pred.cuda()
    print("Evaluating test data...\t test_loader: {}".format(len(test_loader)))
    t1 = time.time()
    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        y_test = torch.cat((y_test, target), 0)
        bs, c, h, w = inp.size()
        # input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
        output = model(input_var)
        # output_mean = output.view(bs, -1).mean(1)
        y_pred = torch.cat((y_pred, output), 0)
        if (i % 10 == 0):
            print("batch: {}".format(i))
    t2 = time.time()
    print("Evaluating time lapse: {} min".format((t2 - t1) // 60))
    
    AUROCs = compute_AUCs(y_test, y_pred)
    # print(AUROCs)
    # print(len(AUROCs))
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

def compute_AUCs(y_test, y_pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        y_test: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        y_pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    y_test_np = y_test.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    # y_pred_np = np.transpose([pred[:, 1] for pred in y_pred_np])
    # print(y_pred_np.shape)
    for i in range(N_CLASSES):
        # print(y_test_np[:, i])
        # print(y_pred_np[:, i])
        result = roc_auc_score(y_test_np[:, i], y_pred_np[:, i])
        AUROCs.append(result)
    return AUROCs

"""Now, let's run"""

# cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). 
# This usually leads to faster runtime.
cudnn.benchmark = True

# initialize and load the model
model = DenseNet121(N_CLASSES).cuda()

"""    if False and os.path.isfile(CKPT_PATH):
  print("=> loading checkpoint")
  checkpoint = torch.load(CKPT_PATH)
  model.load_state_dict(checkpoint['state_dict'])
  print("=> loaded checkpoint")
else:
  print("=> no checkpoint found")
"""

#train_dataset = XrayDataSet(DATA_DIR, "train_val_sample10k.txt")
#test_dataset = XrayDataSet(DATA_DIR, "test_sample1k.txt")

train_dataset = XrayDataSet(DATA_DIR, "labeled_train_val_list.txt")
test_dataset = XrayDataSet(DATA_DIR, "labeled_test_list.txt")

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

train_model(model, train_loader, n_epochs=1)

"""No need to use GPU for calculating AUC"""

gc.collect()
torch.cuda.empty_cache()
# switch to evaluate mode
model.eval()
with torch.no_grad():
  eval_model(model, test_loader)