# -*- coding: utf-8 -*-
"""
将train按类别存放
"""
#OLDDIR = "train"
NEWDIR = "for_train"

import shutil
import os
import pandas as pd

def generate_train(newDir):

    if os.path.exists(newDir):
        shutil.rmtree(newDir)   #清空
    
    f = open('DatasetA_train/train.txt', "r")          #类别信息
    lines = f.readlines()
    for line in lines:
        l = line.strip().split('\t')
        fname = l[0]            #文件名
        breed = l[1]            #类别
        path2 = '%s/%s' % (newDir, breed)
        if not os.path.exists(path2):
            os.makedirs(path2)
        shutil.copyfile('DatasetA_train/train/%s' % fname, '%s/%s' % (path2, fname))   #创立软连接

if __name__ == "__main__":
    generate_train(NEWDIR)      #按类别分类
        
