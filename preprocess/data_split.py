import random
from sklearn.model_selection import train_test_split

random.seed(24)
alldata = list(range(112120))
random.shuffle(alldata)
n_train = int(112120*0.7)
n_val = int(112120*0.8)
train = set(alldata[:n_train])
val = set(alldata[n_train:n_val])
test = set(alldata[n_val:])

f = open("labeled_all.txt", "r")
f_train = open("final_train.txt", "w")
f_val = open("final_val.txt", "w")
f_test = open("final_test.txt", "w")
cnt = 0

for line in f:
    cnt += 1
    if cnt in train:
        f_train.write(line)
    elif cnt in val:
        f_val.write(line)
    else:
        f_test.write(line)

print(cnt)
f.close()
f_train.close()
f_val.close()
f_train.close()