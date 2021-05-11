import cv2
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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

N_LABEL = 14
BATCH_SIZE = 8
DATA_PATH = './images_converted256/'
RANDOM_SEED = 10086
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

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

def collate_fn_train(data):
    image_path, label = zip(*data)
    image_tensors = torch.Tensor()
    trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
                ])
    for img in image_path:
        img_pil = Image.open(img).convert("RGB")
        img_tensor = trans(img_pil).unsqueeze(0)
        image_tensors = torch.cat((image_tensors, img_tensor))
    label_tensors = torch.FloatTensor(label)

    return image_tensors, label_tensors

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

model = DenseNet121(N_LABEL)
# load trained model if needed
model.load_state_dict(torch.load("TrainedModel/ReplicationModel_AUROC_0.8159.pth", map_location = torch.device('cpu')))
model.eval()
# target_layer = model.densenet121.classifier
# target_layer = model.densenet121.features.denseblock1.denselayer6.conv2
# target_layer = model.densenet121.features.denseblock2.denselayer12.conv2
# target_layer = model.densenet121.features.denseblock3.denselayer24.conv2
target_layer = model.densenet121.features.denseblock4.denselayer16.conv2

def reshape_transform(tensor, height=14, width=14):
    print(tensor.shape)
    result = tensor
    # result = tensor.reshape(tensor.size(0), 
    #     height, width, tensor.size(2))
    print(result.shape)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer, reshape_transform=reshape_transform)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = None
# train_dataset = XrayDataSet(DATA_PATH, "train_val_sample1k.txt")
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_train)
# for i, (x, y) in enumerate(train_loader):
#     input_tensor = x
#     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#     grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

#     # In this example grayscale_cam has only one image in the batch:
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_img, grayscale_cam)
image_path = DATA_PATH + "00010351_000.png"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])
print(input_tensor.shape)
# input_tensor = input_tensor.transpose(1, 2).transpose(2, 3)
grayscale_cam = cam(input_tensor=input_tensor,
                    target_category=target_category)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite('cam.jpg', cam_image)
