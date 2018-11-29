#Code to rename the datafiles

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
import pandas as pd

location = 'F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/Data/'


for i in os.listdir(location):
    for j in os.listdir(location + i):

       # j2 = j.split(".")[8]
       # os.rename(location + i +'/' + j, location + i +'/' + j2)
        for k in os.listdir(location+i+'/'+j):
            if 'sheet'  not in k:
                k2 = 'feature_masks'
                os.rename(location + i + '/' + j + '/' + k, location + i + '/' + j + '/' + k2)