#!/usr/bin/env python
# coding: utf-8

# The main CheXNet model implementation.

# In[1]:
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_process import XrayDataSet, collate_fn
from sklearn.metrics import roc_auc_score


# In[17]:
N_CLASSES = 14
CLASS_NAMES = ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", 
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
DATA_DIR = './images_converted/'
TEST_IMAGE_LIST = 'labeled_test_list.txt'
BATCH_SIZE = 16

# In[5]:

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

# In[6]:
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
    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            #print(y.shape)
            #print(y_hat.shape)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss = train_loss / len(train_loader)
        if epoch % 1 == 0:
            train_loss_arr.append(np.mean(train_loss))
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
            #evaluate(model, val_loader)


# In[18]:
    
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

    train_dataset = XrayDataSet(DATA_DIR, "train_val_sample10k.txt")
    test_dataset = XrayDataSet(DATA_DIR, "labeled_test_list.txt")

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    train(model, train_loader)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        #target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


# In[4]:


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs





# In[19]:

if __name__ == '__main__':
    main()