# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:42:55 2016

@author: OustimoA
"""

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nibabel as nib
import cPickle
import pandas as pd

location='Z:/Aimilia/DeepLearningDemo/input_Feb2016/feature maps/'

feature_maps_all={}
for i in os.listdir(location):
    for j in os.listdir(location + i):    
        if 'mask' not in j:
            feature_maps={}            
            for k in os.listdir(location+i+'/'+j):
                if k.endswith(".gz"):
                    feature_map = np.nan_to_num(nib.load(location+i+'/'+j+'/'+k).get_data().T)
                    feature_maps[k.split(".")[0]]=feature_map
    feature_maps_all[i]=feature_maps        
            
masks_all={}
for i in os.listdir(location):
    for j in os.listdir(location + i):    
        if 'mask' in j:
            mask={}            
            for k in os.listdir(location+i+'/'+j):
                if 'sliding' in k:
                    mask = nib.load(location+i+'/'+j+'/'+k).get_data().T
    masks_all[i]=mask        

feature_maps_masked_all={}
for key_1 in feature_maps_all.keys():
    feature_maps_masked={}
    for key_2 in feature_maps_all[key_1].keys():
        feature_map_masked = np.multiply(feature_maps_all[key_1][key_2],masks_all[key_1])
        feature_maps_masked[key_2]=feature_map_masked
    feature_maps_masked_all[key_1]=feature_maps_masked

#item='R_MLO_00830_1_norm_graylevel_win_63_sliding_63_numbin_128_skewness'
#feature_maps_all['R_MLO_00830_1'][item]
#change=np.nan_to_num(feature_maps_all['R_MLO_00830_1'][item])
#change

#feature_maps_masked_all.keys()
#feature_maps_masked_all['R_MLO_20527_1']
#feature_maps_masked_all['L_MLO_00751_1']
#


feat_names=[]
for key in feature_maps_masked_all['L_MLO_00751_1'].keys():
    feat_names.append(key.split("_norm_")[1])

###############################################################################
#start code snippet
#code for normalizing over all samples for each feature map

fsep={}
max_abs_all={}
min_all={}
max_all={}    
for name in feat_names:
    flist=[]
    max_abs_list=[]
    max_list=[]
    min_list=[]    
    for key_1 in feature_maps_masked_all.keys():
        flist.append(feature_maps_masked_all[key_1][key_1+'_norm_'+name])
        max_abs_list.append(np.max(np.abs(feature_maps_masked_all[key_1][key_1+'_norm_'+name])))
        min_list.append(np.min(feature_maps_masked_all[key_1][key_1+'_norm_'+name]))
        max_list.append(np.max(feature_maps_masked_all[key_1][key_1+'_norm_'+name]))
    fsep[name]=flist
    max_abs_all[name]=max_abs_list
    min_all[name]=min_list
    max_all[name]=max_list
    

# the histogram of the data
check=max_abs_all[feat_names[29]]
plt.hist(check,normed=1)

n, bins, patches = plt.hist(max_abs_all[feat_names[0]], 20, normed=1, facecolor='green', alpha=0.75)
plt.hist(max_abs_all[max_abs_all.keys()[0]], 20, normed=1, facecolor='green', alpha=0.75)
plt.grid(True)
plt.show()
        
max_dict={}
for key in fsep.keys():
    max_dict[key]=np.max(np.abs(np.round(fsep[key])))
    
epsilon = 0.00000001
f_masked_normed_all={}
for key_1 in feature_maps_masked_all.keys():
    f_masked_normeds={}
    for key_2 in feature_maps_masked_all[key_1].keys():
        f_masked_normed = feature_maps_masked_all[key_1][key_2]/(max_dict[key_2.split('_norm_')[1]]+epsilon)
        f_masked_normeds[key_2]=f_masked_normed
    f_masked_normed_all[key_1]=f_masked_normeds

#end code snippet
###############################################################################
###############################################################################
# start code snippet
#code for normalizing for each sample's feature map





epsilon = 0.00000001
f_masked_normed_all={}
for key_1 in feature_maps_masked_all.keys():
    f_masked_normeds={}
    for key_2 in feature_maps_masked_all[key_1].keys():
        f_masked_normed = feature_maps_masked_all[key_1][key_2]/(np.abs(np.max(feature_maps_masked_all[key_1][key_2])+epsilon))
        f_masked_normeds[key_2]=f_masked_normed
    f_masked_normed_all[key_1]=f_masked_normeds

#end code snippet
###############################################################################

f_masked_normed_all['R_MLO_20527_1']['R_MLO_20527_1_norm_cooccurrence_win_63_sliding_63_numbin_128_offset_11_entropy']
test_img=f_masked_normed_all['R_MLO_20527_1']['R_MLO_20527_1_norm_cooccurrence_win_63_sliding_63_numbin_128_offset_11_entropy']
plt.imshow(test_img)

f=open('Z:/Andrew/deep_learning/dicts/features_36_by_30_3_normed_ineachsample.p','w')
cPickle.dump(f_masked_normed_all,f)
f.close()



#f=open('Z:/Andrew/deep_learning/dicts/features_36_by_30_3_normed_ineachsample.p','r')
#check=cPickle.load(f)
#f.close()

#check if all feature maps are the same size
#ydim=[]
#xdim=[]
#for key_1 in f_masked_normed_all.keys():
#    shapey_list=[]
#    shapex_list=[]    
#    for key_2 in f_masked_normed_all[key_1].keys():
#        shapey=f_masked_normed_all[key_1][key_2].shape[0]
#        shapex=f_masked_normed_all[key_1][key_2].shape[1]
#        shapey_list.append(shapey)
#        shapex_list.append(shapex)
#    ydim.extend(shapey_list)
#    xdim.extend(shapex_list)    
#
#y=36*len(ydim)
#y==np.sum(ydim)
#x=30*len(xdim)
#x==np.sum(xdim)

#importing case-control status
cc_status=pd.read_excel('Z:/Aimilia/DeepLearningDemo/input_Feb2016/case_control_status.xlsx',0)
cc_status

target_dict={}
for i in np.arange(len(cc_status.ix[:,0])):
    target_dict[str(cc_status.ix[i,0])]=cc_status.ix[i,2]
    
#f=open('Z:/Andrew/deep_learning/dicts/target_dict.p','w')
#cPickle.dump(target_dict,f)

feat_list=[]
target_list=[]
for key_1 in target_dict.keys():
    feat_list.append(f_masked_normed_all[key_1])
    target_list.append(target_dict[key_1])
    
data=zip(feat_list,target_list)

f=open('Z:/Andrew/deep_learning/dicts/data_combo_3_normed_ineachsample.p','w')
cPickle.dump(data,f)
f.close()

#f=open('Z:/Andrew/deep_learning/dicts/data_combo_3_normed_ineachsample.p','r')
#check=cPickle.load(f)
#f.close()