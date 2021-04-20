import random
from sklearn.model_selection import train_test_split

random.seed(10086)

f = open("labeled_all.txt", "r")
patient = {}
for line in f:
    img = line.split(" ")[0]
    pid = int(img.split("_")[0])
    if pid in patient.keys():
        patient[pid] += 1
    else:
        patient[pid] = 1
f.close()
allpid = list(patient.keys())
random.shuffle(allpid)

group = {k: 0 for k in patient.keys()}

tot = 0
curgroup = 1
for pid in allpid:
    group[pid] = curgroup
    tot += patient[pid]
    if tot>112120*0.7 and tot<=112120*0.8:
        curgroup = 2
    elif tot>112120*0.8:
        curgroup = 3

f = open("labeled_all.txt", "r")
f_train = open("final_train.txt", "w")
f_val = open("final_val.txt", "w")
f_test = open("final_test.txt", "w")
cnt = 0

for line in f:
    cnt += 1
    img = line.split(" ")[0]
    pid = int(img.split("_")[0])
    if group[pid]==1:
        f_train.write(line)
    elif group[pid]==2:
        f_val.write(line)
    else:
        f_test.write(line)

print(cnt)
f.close()
f_train.close()
f_val.close()
f_train.close()