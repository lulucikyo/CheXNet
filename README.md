# CheXNet
[CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) Replication and Improvment experiments for *CS 598 Deep Learning for Healthcare*

This project took [ChestX-ray 14](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) image dataset and predicted probabilities for 14 types of chest diesease.

## Dataset
The ChestX-ray 14 dataset contains 112,120 chest X-ray images of 30,805 unique patients with 14 disease labels. As per the original work, we roughly split the dataset into training set (70%), validation set (10%) and test set (20%), with no patient overlaps between dataset partitions. 

## Directory Structure
/preprocess:
  - data_label.py - convert labels into multihot label vector
  - data_resize.py - resize original images
  - data_split.py - partition dataset into train, validation, test
  - data_unzip.py - automate dataset tarball unzip
  - <span>sample.py</span> - take sample data for small scale test

final_test.txt - test set: filename, label vector

final_train.txt - train set: filename, label vector

final_val.txt - validation set: filename, label vector

***replication_chexnet_cuda_local\*.py - the main program***

## Experiment designs

Model Component | Variants
---|---
preprocess step | option 1(default): resize to 224×224 with normalized based on ImageNet
data augmentation | option 1: raw (224×224) only<br>option 2(default): raw (224×224) with random horizontal flip<br>option 3: raw (256×256) with (horizontally flip + randomly crop) (limit crop size to (224×224))
backbone | option 1(default): DenseNet121<br>option 2: MobileNetV2<br>option 3:MobileNetV3-Large<br>option 4: DenseNet169<br>option 5: ResNet18<br>
batch size | option 1(default): 16<br>option 2: 32<br>option 3: 64<br>
Initial Weights | option 1(default): ImageNet
optimizer | option 1(default): Adam (1 = 0.9 and 2 = 0.999)
Initial learning rate | option 1(default): initial value =  0.001<br>option 2: initial value = 0.01<br>option 3: initial value = 0.0005<br>option 4: initial value = 0.0001<br>option 5: initial value = 0.00005<br>
learning rate<br>decay factor | option1(default): 10



## Model comparison
| | Wang et al. (2017) | Yao et al. (2017) | CheXNet | Our Best Model |
|---|:---:|:---:|:---:|:---:|
Atelectasis	| 0.716 |	0.772	| 0.8094 |	0.8274
Cardiomegaly |	0.807 |	0.904 |	0.9248 |	0.9130
Effusion |	0.784 |	0.859 |	0.8638 |	0.8799
Infiltration |	0.609 |	0.695 |	0.7345 |	0.7181
Mass |	0.706 |	0.792 |	0.8676 |	0.8667
Nodule |	0.671 |	0.717 |	0.7802	| 0.7931
Pneumonia |	0.633 |	0.713 |	0.768 |	0.7414
Pneumothorax |	0.806 |	0.841 |	0.8887 |	0.8886
Consolidation |	0.708 |	0.788 |	0.7901 |	0.8269
Edema |	0.835 |	0.882 |	0.8878 |	0.8848
Emphysema |	0.815 |	0.829 |	0.9371 |	0.9336
Fibrosis |	0.769 |	0.767 |	0.8047 |	0.8194
Pleural_Thickening |	0.708 |	0.765 |	0.8062 |	0.8115
Hernia |	0.767 |	0.914 |	0.9164 |	0.9198
Average AUROC |	0.738 |	0.803 |	0.8414 |	0.8446
## Prerequisite
- Python 3.7+
- PyTorch 1.8.1
- Numpy
- sklearn

## Usage
1. Download the dataset (`/images`), dataset partition list (`train_val_list.txt`, `test_list.txt`) and labels (`Data_Entry_2017_v2020.csv`) from [ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) (find the [README](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610) file in it helpful)
2. Unzip the tarballs using `data_unzip.py`
3. (Optional) Resize the images using `data_resize.py`
4. Split dataset into `train`, `validation`, `test` set using `data_split.py`
5. Generate `(filename, label vector)` tuple list by `data_label.py`
6. Specify image folder path, i.e.
```
DATA_PATH = './images_converted256/'
```
7. Run
```
python replication_chexnet_cuda_local.py
```
## Contributor
Xi Li, Yu Liu, Liping Xie, Yekai Yu


