#Code to rename the datafiles

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
import pandas as pd

# Replace the location value where you have the data
location = '/Users/paulhsu/CIS537_Data/'
		
for i in os.listdir(location):

	# Skip '.DS_Store' file in mac os, which is not a directory
	if ".DS_Store" in i:
		continue
		
	for j in os.listdir(location + i):
		# j2 = j.split(".")[8]
		# os.rename(location + i +'/' + j, location + i +'/' + j2)
		
		# Skip '.DS_Store' file in mac os, which is not a directory
		if ".DS_Store" in j:
			continue
		for k in os.listdir(location+i+'/'+j):

			# Skip '.DS_Store' file in mac os, which is not a directory
			if ".DS_Store" in k:
				continue
			if 'sheet'  not in k:
				k2 = 'feature_masks'
				os.rename(location + i + '/' + j + '/' + k, location + i + '/' + j + '/' + k2)