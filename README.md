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
[placeholder]

## Result comparison
[placeholder]

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


