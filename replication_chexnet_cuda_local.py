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

N_LABEL = 14
LABELS = ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", 
          "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
          "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
BATCH_SIZE = 8
N_EPOCH = 16
PRINT_INTERVAL = 100
RANDOM_SEED = 10086
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
TIME_STAMP = time.strftime("%m-%d-%Y-%H-%M-%Z", time.localtime())
DATA_PATH = './images_converted256/'
MODEL_PATH = './saved_model/' + time.strftime("%m-%d-%Y", time.localtime()) + "/b" + str(BATCH_SIZE) + "e" + str(N_EPOCH) + "/"
LOG_PATH = './log/'

# assume we will take 256 * 256 as input
# so that we can do crop operations at a later point of time
def collate_fn(data):
    image_path, label = zip(*data)
    image_tensors = torch.Tensor()
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225]),
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([(crop) for crop in crops]))
                ])
    for img in image_path:
        img_pil = Image.open(img).convert("RGB")
        img_tensor = trans(img_pil).unsqueeze(0)
        image_tensors = torch.cat((image_tensors, img_tensor))
    label_tensors = torch.FloatTensor(label)

    return image_tensors.cuda(), label_tensors.cuda()

class XrayDataSet(Dataset):
    def __init__(self, data_path, image_list):
        self.image_path = []
        self.y = []
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
    """
    The last layer of DenseNet121 was replaced by a Linear with 14 output features, followed by a sigmoid function
    """
    def __init__(self, out_feature):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features, out_feature),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def train_model(model, train_loader, val_loader, n_epochs, logfile):
    t1 = time.time()
    criterion = nn.BCELoss()
    """using Adam with standard parameters (B1 = 0.9 and B2 = 0.999) """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    """factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1."""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                               patience=1, verbose=False, threshold=0.0001,
                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    # prep model for training
    model.train()

    train_loss_arr = []

    log = open(logfile, "a+")
    log.write("Started training, total epoch : {}\n".format(n_epochs))
    log.write("Training data size: {}\n".format(len(train_loader)))
    print("Started training, total epoch : {}\n".format(n_epochs))
    print("Training data size: {}\n".format(len(train_loader)))
    
    for epoch in range(n_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = 0
        batch = 0
        log.write("Started epoch {}\n".format(epoch+1))
        print("Started epoch {}\n".format(epoch+1))
        for x, y in train_loader:
            optimizer.zero_grad()
            batches, ncrops, channel, height, width = x.size()
            with torch.no_grad():
                x_in = torch.autograd.Variable(x.view(-1, channel, height, width).cuda())
            y_hat = model(x_in)
            y_hat = y_hat.view(batches, ncrops, -1).mean(1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if ((batch+1) % PRINT_INTERVAL == 0):
                log.write('Trained {} batches \tTraining Loss: {:.6f}\n'.format(batch+1, loss.item()))
                print('Trained {} batches \tTraining Loss: {:.6f}\n'.format(batch+1, loss.item()))
            batch += 1

        train_loss = train_loss / len(train_loader)
        
        train_loss_arr.append(np.mean(train_loss))
        log.write('Epoch: {} \tTraining Loss: {:.6f}\n'.format(epoch+1, train_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}\n'.format(epoch+1, train_loss))
        torch.save(model.state_dict(), MODEL_PATH + str(epoch)+"trained.pth")

        model.eval()
        log.write('AUROCs on validation dataset:\n')
        log.close()
        val_loss = 0
        with torch.no_grad():
            val_loss = eval_model(model, val_loader, logfile)
        log = open(logfile, "a+")
        log.write('Epoch: {} \tLearning Rate for first group: {:.10f}\n'.format(epoch+1, optimizer.param_groups[0]['lr']))
        model.train()
        scheduler.step(val_loss)

    t2 = time.time()
    log.write("Training time lapse: {} min\n".format((t2 - t1) // 60))
    print("Training time lapse: {} min\n".format((t2 - t1) // 60))
    log.close()

def eval_model(model, test_loader, logfile):
    # initialize the y_test and y_pred tensor
    log = open(logfile, "a+")

    y_test = torch.FloatTensor()
    y_test = y_test.cuda()
    y_pred = torch.FloatTensor()
    y_pred = y_pred.cuda()
    criterion = nn.BCELoss()
    val_loss = 0
    log.write("Evaluating test data...\t test_loader: {}\n".format(len(test_loader)))
    print("Evaluating test data...\t test_loader: {}\n".format(len(test_loader)))
    t1 = time.time()
    for i, (x, y) in enumerate(test_loader):
        y = y.cuda()
        y_test = torch.cat((y_test, y), 0)
        batches, ncrops, channel, height, width = x.size()
        with torch.no_grad():
            x_in = torch.autograd.Variable(x.view(-1, channel, height, width).cuda())
        y_hat = model(x_in)
        y_hat = y_hat.view(batches, ncrops, -1).mean(1)
        y_pred = torch.cat((y_pred, y_hat), 0)
        loss = criterion(y_hat, y)
        val_loss = val_loss + loss.item()
        if (i % PRINT_INTERVAL == 0):
            log.write("batch: {}\n".format(i))
            print("batch: {}".format(i))
    t2 = time.time()
    val_loss = val_loss / len(test_loader)
    log.write("Evaluating time lapse: {} min\n".format((t2 - t1) // 60))
    print("Evaluating time lapse: {} min\n".format((t2 - t1) // 60))
    log.write('The total val loss is {val_loss:.7f}\n'.format(val_loss=val_loss))

    """Compute AUROC for each class"""
    AUROCs = []
    y_test_np = y_test.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    for i in range(N_LABEL):
        result = roc_auc_score(y_test_np[:, i], y_pred_np[:, i])
        AUROCs.append(result)

    AUROC_avg = np.array(AUROCs).mean()
    log.write('The average AUROC is {AUROC_avg:.3f}\n'.format(AUROC_avg=AUROC_avg))
    print('The average AUROC is {AUROC_avg:.3f}\n'.format(AUROC_avg=AUROC_avg))
    for i in range(N_LABEL):
        log.write('The AUROC of {} is {}\n'.format(LABELS[i], AUROCs[i]))
        print('The AUROC of {} is {}\n'.format(LABELS[i], AUROCs[i]))

    log.close()
    return val_loss



"""Now, let's run"""

# CUDA stats
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). 
# This usually leads to faster runtime.
cudnn.benchmark = True

# initialize and load the model
model = DenseNet121(N_LABEL).cuda()

# Small sample for debug purpose. Commented out for full training

# train_dataset = XrayDataSet(DATA_PATH, "train_val_sample1k.txt")
# test_dataset = XrayDataSet(DATA_PATH, "test_sample1k.txt")


# train_dataset = XrayDataSet(DATA_PATH, "labeled_train_val_list.txt")
# test_dataset = XrayDataSet(DATA_PATH, "labeled_test_list.txt")


train_dataset = XrayDataSet(DATA_PATH, "final_train.txt")
val_dataset = XrayDataSet(DATA_PATH, "final_val.txt")
test_dataset = XrayDataSet(DATA_PATH, "final_test.txt")

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(len(train_loader), len(val_loader), len(test_loader))

logfile = LOG_PATH + TIME_STAMP + "-runlog.txt"
try:
    if (os.path.exists(LOG_PATH)):
        print(LOG_PATH + "\nalready exists, go on.\n")
    else:
        os.makedirs(LOG_PATH, exist_ok = True)
except FileExistsError:
    print(LOG_PATH + "\nalready exists, go on.\n")

try:
    if not (os.path.exists(MODEL_PATH)):
        os.makedirs(MODEL_PATH, exist_ok = True)
        print("Creating " + MODEL_PATH + "\n")
    else:
        print(MODEL_PATH + "\nalready exists, go on.\n")
except FileExistsError:
    print(MODEL_PATH + "\nalready exists, go on.\n")
gc.collect()
torch.cuda.empty_cache()
train_model(model, train_loader, val_loader, N_EPOCH, logfile)

"""No need to use GPU for calculating AUC"""
gc.collect()
torch.cuda.empty_cache()
# switch to evaluate mode
model.eval()
with torch.no_grad():
    eval_model(model, test_loader, logfile)
