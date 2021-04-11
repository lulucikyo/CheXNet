# get 10% data from labeled_train_val_list.txt
import random

f = open("labeled_train_val_list.txt", "r")
alldata = f.readlines()
f.close()
sample = random.sample(alldata, 1000)
print(len(alldata))

f = open("train_val_sample1k.txt", "w")
f.writelines(sample)
f.close()