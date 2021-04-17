import os
import torchvision
import torchvision.transforms as transforms

from PIL import Image



DATA_PATH = "../images/"
NEW_PATH = "../images_converted256/"
img_list = os.listdir(DATA_PATH)
converted_list = set(os.listdir(NEW_PATH))
#test_img = ['00000001_000.png', '00000001_001.png', '00000001_002.png', '00000002_000.png', '00000003_000.png', '00000003_001.png', '00000003_002.png', '00000003_003.png', '00000003_004.png', '00000003_005.png']
trans = transforms.Compose([
                transforms.Resize(256)
                ])

cnt = len(converted_list)
for img in img_list:
    if img in converted_list:
        continue

    img_tensor = Image.open(DATA_PATH+img).convert("RGB")
    img_cvt = trans(img_tensor)
    img_cvt.save(NEW_PATH+img)
    cnt += 1
    if cnt % 1000 ==0:
        print("Converted "+str(cnt)+" images")