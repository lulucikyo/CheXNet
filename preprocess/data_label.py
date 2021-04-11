import pandas as pd
from collections import defaultdict

LABEL_FILE = "Data_Entry_2017_v2020.csv"

LABELS =["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", 
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

def label_process(list_file, label_df, label_dict):
        f = open(list_file, "r")
        fw = open("labeled_"+list_file, "w")
        for line in f:
            image = line.strip("\n")
            lb = label_df[label_df["Image Index"]==image]["Finding Labels"]
            #print(lb.values[0])
            lb_word = lb.values[0].split("|")
            ans=["0" for i in range(14)]
            if lb_word[0]!="No Finding":
                for word in lb_word:
                    ans[label_dict[word]] = "1"
            res = image+" "+" ".join(ans)+"\n"
            fw.write(res)
        f.close()
        fw.close()


label_dict = dict(zip(LABELS, range(14)))
label_df = pd.read_csv(LABEL_FILE)
df = pd.DataFrame(label_df, columns=["Image Index", "Finding Labels"])
df = df.convert_dtypes()
#label_df["Image Index"] = label_df["Image Index"].astype("str")
#label_df["Finding Labels"] = label_df["Finding Labels"].astype("str")
#print(df.head(5), df.dtypes)
#print(df[df["Image Index"]=="00000001_000.png"])

label_process("test_list.txt", df, label_dict)
label_process("train_val_list.txt", df, label_dict)