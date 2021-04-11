#!/usr/bin/env python
# coding: utf-8

# The main CheXNet model implementation.

# In[1]:
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_process import XrayDataSet, collate_fn
from sklearn.metrics import roc_auc_score


N_CLASSES = 14
CLASS_NAMES = ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", 
               "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
               "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
DATA_DIR = './images_converted/'
TEST_IMAGE_LIST = 'labeled_test_list.txt'
BATCH_SIZE = 16


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

def train(model, train_loader, n_epochs = 1):

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
            if (batch % 10 == 0):
                print('Trained {} batches \tTraining Loss: {:.6f}'.format(batch, loss.item()))
            batch += 1

        train_loss = train_loss / len(train_loader)
        scheduler.step(train_loss)
        if epoch % 1 == 0:
            train_loss_arr.append(np.mean(train_loss))
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
            #evaluate(model, val_loader)

    
def main():
    # cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES)

    """    if False and os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
    """

    train_dataset = XrayDataSet(DATA_DIR, "train_val_sample1k.txt")
    test_dataset = XrayDataSet(DATA_DIR, "test_sample1k.txt")

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    t1 = time.time()
    train(model, train_loader)
    t2 = time.time()
    print("Training time lapse: {} min".format((t2 - t1) // 60))
    # initialize the ground truth and output tensor
    y_test = torch.FloatTensor()
    # y_test = y_test.cuda()
    y_pred = torch.FloatTensor()
    # y_pred = y_pred.cuda()

    # switch to evaluate mode
    model.eval()

    print("Evaluating test data...\t test_loader: {}".format(len(test_loader)))
    t1 = time.time()
    for i, (inp, target) in enumerate(test_loader):
        #target = target.cuda()
        y_test = torch.cat((y_test, target), 0)
        # print(target.shape)
        # print(y_test.shape)
        # print(inp)
        # print(inp.size())
        bs, c, h, w = inp.size()
        # input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w))
        output = model(input_var)
        # print(output.shape)
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
    y_test_np = y_test.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    # y_pred_np = np.transpose([pred[:, 1] for pred in y_pred_np])
    # print(y_pred_np.shape)
    for i in range(N_CLASSES):
        # print(y_test_np[:, i])
        # print(y_pred_np[:, i])
        result = roc_auc_score(y_test_np[:, i], y_pred_np[:, i])
        AUROCs.append(result)
    return AUROCs

if __name__ == '__main__':
    main()