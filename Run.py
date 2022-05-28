import os
from model_origin import *
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import time
import warnings

Model_name='model_4layers_22/billy_train_201.pth'
warnings.filterwarnings("ignore")
root = tk.Tk()
root.withdraw()
#打开文件
file_path = filedialog.askopenfilename(filetypes=[('jpg or csv files', '.csv .jpg .jpeg')])
def ReadJPG(path):
        img = Image.open(path)
        img = img.resize((100,100))
        trans = transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor()
                #transforms.Resize(32,32)
                ]
        )
        img = trans(img)
        img = torch.unsqueeze(img,0)
        return img
def ReadCSV(path):
        Raw = pd.read_csv(path, encoding="gbk")
        DischargeCOL = Raw.loc[Raw[r'工步状态'].str.contains('RateD|CCD')].values.T
        a = max(DischargeCOL[0])
        b = 4
        for x in range(len(DischargeCOL[0])):
                DischargeCOL[0][x] = DischargeCOL[0][x] / a
                DischargeCOL[1][x] = DischargeCOL[1][x] / b
        plt.rcParams['savefig.dpi'] = 100
        plt.figure(figsize=(1, 1), facecolor='black')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.plot(DischargeCOL[0], DischargeCOL[1], color='w')
        plt.savefig('cache/tem.jpg')
        plt.close()
        img = Image.open('cache/tem.jpg')
        img = img.resize((100, 100))
        trans = transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor(),
                # transforms.Resize(32,32)
        ]
        )
        img = trans(img)
        img = torch.unsqueeze(img, 0)
        return img

form = os.path.splitext(file_path)[-1][1:]
if form == "csv":
        img1 = ReadCSV(file_path)
elif form == "jpg":
        img1 = ReadJPG(file_path)
model_100 = torch.load(Model_name)
start_time = time.time()
result = model_100(img1).argmax(1).numpy().tolist()
end_time = time.time()
label = {"0":"--------------------------------------------------\n\n---------------------单平台LSB---------------------\n\n--------------------------------------------------",
         "1":"--------------------------------------------------\n\n------------------双平台液相转化LSB------------------\n\n--------------------------------------------------",
         "2":"--------------------------------------------------\n\n------------极化过严重放不出第二个平台的LSB-------------\n\n--------------------------------------------------",
         "3":"--------------------------------------------------\n\n---------无法形成有效CEI层 固相反应转化失败的LSB--------\n\n--------------------------------------------------",
         "4": "-------------------------------------------------\n\n----------------锂离子电池体系(LFP)-------------------\n\n--------------------------------------------------",
         "5": "-------------------------------------------------\n\n----------------锂离子电池体系(NCM)-------------------\n\n--------------------------------------------------",
         }
if result == [0]:
        print('-----------------固相转化锂硫电池--------------------')
else:
        print("------------------非固相转化锂硫电池------------------")
print(label[str(result[0])])

print('---------time_used: {}----------'.format(end_time - start_time))
#input("--------------------按Enter退出-----------------------")
