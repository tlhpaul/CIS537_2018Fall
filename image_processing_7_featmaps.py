import os
import numpy as np
import nibabel as nib
import _pickle as pickle
import pandas as pd

location = '/Users/paulhsu/CIS537_Data/'

feature_maps_all={}
masks_all={}
count2 = 0

#10 minutes for entire dataset on the loop below
#below code storing only left image
for i in os.listdir(location):  #user id
	print('Patient id : ', i)
    # Skip '.DS_Store' file in mac os, which is not a directory
	if ".DS_Store" in i:
		continue
	#left and right
	for j in os.listdir(location + i):
    	# Skip '.DS_Store' file in mac os, which is not a directory
		if ".DS_Store" in j:
			continue
		for k in os.listdir(location+i+'/'+j): #feature_masks and sheet
        	
        	# Skip '.DS_Store' file in mac os, which is not a directory
			if ".DS_Store" in k:
				continue
			if 'sheet'  not in k:
				for l in os.listdir(location+i+'/'+j+'/'+k):  #feature and masks
					if 'mask' not in l:
						feature_maps = {}
						count = 0

						# Skip '.DS_Store' file in mac os, which is not a directory
						if ".DS_Store" in l:
							continue						
						for m in os.listdir(location+i+'/'+j+'/'+k+'/'+l): #feature
							if m.endswith(".gz"):
                                #print(m)
								loc = (location + i + '/' + j + '/' + k + '/' + l + '/' + m)
								feature_map = np.nan_to_num(nib.load(loc).get_data().T)
								feature_maps[count]= feature_map
								count = count + 1
					else:
						mask = {}
						for n in os.listdir(location + i + '/' + j + '/' + k + '/' + l):

							if n.endswith(".gz"):
								#print(n)
								mask = nib.load(location + i + '/' + j + '/' + k + '/' + l + '/' + n).get_data().T
	count2 = count2 + 1
	if count2 == 400:
		break
	masks_all[i] = mask


	feature_maps_all[i]=feature_maps


#MULTIPLYING THE FEATURES AND MASKS
feature_maps_masked_all={}
for key_1 in feature_maps_all.keys():
    print(key_1)
    feature_maps_masked={}
    for key_2 in feature_maps_all[key_1].keys():
        feature_map_masked = np.multiply(feature_maps_all[key_1][key_2],masks_all[key_1])
        feature_maps_masked[key_2]=feature_map_masked
    feature_maps_masked_all[key_1]=feature_maps_masked

# FINDING MIN AND MAX FOR EACH FEATURE ACROSS ALL EXAMPLES(NOT BEING USED)
# Normalizing over all samples for each feature map
fsep={}
max_abs_all={}
min_all={}
max_all={}    
for i in range(29):  #over all different features
    flist=[] #making a list across all ids of feature type
    max_abs_list=[] #making a list across all ids of absolute maximum for each feature type
    max_list=[]
    min_list=[]    
    for key_1 in feature_maps_masked_all.keys():
        flist.append(feature_maps_masked_all[key_1][i])
        max_abs_list.append(np.max(np.abs(feature_maps_masked_all[key_1][i])))
        min_list.append(np.min(feature_maps_masked_all[key_1][i]))
        max_list.append(np.max(feature_maps_masked_all[key_1][i]))
    fsep[i]=flist
    max_abs_all[i]=max_abs_list
    min_all[i]=min_list
    max_all[i]=max_list

# max_dict={}
# for key in fsep.keys():
#     max_dict[key]=np.max(np.abs(np.round(fsep[key])))
# epsilon = 0.00000001
# f_masked_normed_all={}
# for key_1 in feature_maps_masked_all.keys():
#     f_masked_normeds={}
#     for key_2 in feature_maps_masked_all[key_1].keys():
#         f_masked_normed = feature_maps_masked_all[key_1][key_2]/(max_dict[key_2.split('_norm_')[1]]+epsilon)
#         f_masked_normeds[key_2]=f_masked_normed
#     f_masked_normed_all[key_1]=f_masked_normeds

# Normalizing for each sample's feature map
epsilon = 0.00000001
f_masked_normed_all={}
for key_1 in feature_maps_masked_all.keys():
    f_masked_normeds={}
    for key_2 in feature_maps_masked_all[key_1].keys():
        f_masked_normed = feature_maps_masked_all[key_1][key_2]/(np.abs(np.max(feature_maps_masked_all[key_1][key_2])+epsilon))
        f_masked_normeds[key_2]=f_masked_normed
    f_masked_normed_all[key_1]=f_masked_normeds


# Importing case-control status
cc_status=pd.read_excel('controlcase.xlsx', 0)

target_dict={}
for i in np.arange(len(cc_status.ix[:,0])):
    target_dict[str(cc_status.ix[i,3])]=cc_status.ix[i,1]


feat_list=[]
target_list=[]
for key_1 in feature_maps_masked_all.keys():
    feat_list.append(f_masked_normed_all[key_1])
    target_list.append(target_dict[key_1])

data=zip(feat_list,target_list)

savePath = './processed_data.p'

#The output file needs to be opened in binary mode:In Python 3, Binary modes 'wb', 'rb' must be specified whereas in Python 2x, they are not needed
f=open(savePath,'wb')
pickle.dump(data,f)
f.close()

