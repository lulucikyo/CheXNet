# CheXNet
[CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) Replication and Improvment experiments for *CS 598 Deep Learning for Healthcare*

This project took [ChestX-ray 14](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) image dataset and predicted probabilities for 14 types of chest diesease.

# Dataset

The ChestX-ray 14 dataset contains 112,120 chest X-ray images of 30,805 unique patients with 14 disease labels. As per the original work, we roughly split the dataset into training set (70%), validation set (10%) and test set (20%), with no patient overlaps between dataset partitions. 

## Directory Structure
---
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

## Data File:
---
### Original Files
- train_val_list.txt - original train & validation image list
- test_list.txt - original test image list 
### Labeled Files
- labeled_train_val_list.txt - tran_val_list.txt combined with multi-hot labels of 14 diseases
- labeled_test_list.txt - test_list.txt combined with multi-hot labels of 14 diseases
### Sampled Files
- train_val_sample1k.txt - contains 1k images from labeled_train_val_list.txt
- train_val_sample10k.txt - contains 10k images from labeled_train_val_list.txt

## Experiment designs
---
[placeholder]

## Result comparison
---
[placeholder]

## Contributor
Xi Li, Yu Liu, Liping Xie, Yekai Yu


