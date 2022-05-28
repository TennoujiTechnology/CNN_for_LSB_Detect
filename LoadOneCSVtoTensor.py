# -*- coding: UTF-8 -*-
import pandas as pd
import os
from tkinter import filedialog
from matplotlib import pyplot as plt

def createIMG(DischargeCOL,out_path):
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

def readCSV(filepath='.',out_path="1",cycles=250):
    Raw = pd.read_csv(filepath,encoding="gbk")
    ALLDischargeCOL = Raw.loc[Raw[r'工步状态'].str.contains('RateD|CCD')]
    for x in range(cycles):
        SpecificDischargeCOL = ALLDischargeCOL.loc[ALLDischargeCOL[r'循环序号']==x+1].values.T
        createIMG(SpecificDischargeCOL,out_path+str(x)+'.jpg')








if __name__ == '__main__':
    readCSV('cache/200cycles.csv','cache/outputs',250)
