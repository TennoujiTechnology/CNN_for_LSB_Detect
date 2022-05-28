# -*- coding: UTF-8 -*-
import pandas as pd
import os
from tkinter import filedialog
from matplotlib import pyplot as plt

#读取csv
def readCSV(filepath='.',out_path="1"):
    Raw = pd.read_csv(filepath,encoding="gbk")
    DischargeCOL = Raw.loc[Raw[r'工步状态'].str.contains('RateD|CCD')].values.T
    if len(DischargeCOL[0]):
        a = max(DischargeCOL[0])
        b = 4
        for x in range(len(DischargeCOL[0])):
            DischargeCOL[0][x] = DischargeCOL[0][x] / a
            DischargeCOL[1][x] = DischargeCOL[1][x] / b
        plt.rcParams['savefig.dpi'] = 100
        plt.figure(figsize=(1, 1),facecolor='black')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.plot(DischargeCOL[0],DischargeCOL[1],color='w')
        plt.savefig(out_path)
        plt.close()
        return

def Run(tem):
    for path,dir_list,file_list in tem:
        for f in file_list:
            if os.path.splitext(f)[-1] == '.csv':
                if os.path.getsize(os.path.join(path,f)) > 10:
                    print(os.path.join(path,f),os.path.getsize(os.path.join(path,f)))
                    name = os.path.splitext(f)[0]+'.jpg'
                    readCSV(os.path.join(path,f),os.path.join(path,name))

if __name__ == '__main__':
    dir = filedialog.askdirectory(initialdir=os.getcwd(),title='select a folder')
    tem = os.walk(dir)
    Run(tem)