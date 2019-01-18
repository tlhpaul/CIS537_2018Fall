
import _pickle as pickle
import numpy as np
import tensorflow as tf
from random import shuffle

f=open('F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/dicts/data_norm_all.p','rb')
data=pickle.load(f)
f.close()

data = list(data)


feature_label =[]
features={}
labels=[]
count = 0

indexcontrol = []
indexcase = []

for i in range(len(data)):
     if data[i][1] == 0: #control
         indexcontrol.append(i)
     else:
         indexcase.append(i)

#sample 200 cases(repeat and randomly shuffle) and 200 controls

indexcontroltrain = indexcontrol[0:200] #200 controls in train
indexcasetrain = indexcase[0:60] #60 cases for replicating in test

indexcontroltest = indexcontrol[200:] #260 controls remaining in test
indexcasetest = indexcase[60:] #114-60 = 54 cases in test

indexcasetrain = [*indexcasetrain , *indexcasetrain, *indexcasetrain, *indexcasetrain] #duplicate cases
shuffle(indexcasetrain)
indexcasetrain = indexcasetrain[0:200] # 200 cases in test
index = [*indexcontroltrain , *indexcasetrain] #400 cases+controls in train
shuffle(index)

indextest = [*indexcontroltest , *indexcasetest] #260 controls + 54 cases in test , 260/54 = 4.8
shuffle(indextest)


features = np.zeros((400,29,34,26))
for i in range(400):
    feat_dict = data[index[i]][0]
    labels.append(data[index[i]][1])
    for key2 in feat_dict.keys():
        p = feat_dict[key2]
        features[count, key2, :, :] = p[0:34,0:26]
    count = count + 1


featurestest = np.zeros((len(indextest),29,34,26))
labelstest = []
count = 0
for i in range(len(indextest)):
    feat_dict = data[indextest[i]][0]
    labelstest.append(data[indextest[i]][1])
    for key2 in feat_dict.keys():
        p = feat_dict[key2]
        featurestest[count, key2, :, :] = p[0:34,0:26]
    count = count + 1

#34 X 26 is the common image size, some are 42 X 37 (need to crop central part for such cases- change this)
# features = np.zeros((len(data),29,34,26))
# for i in range(len(data)):
#     feat_dict = data[i][0]
#     labels.append(data[i][1])
#     for key2 in feat_dict.keys():
#         p = feat_dict[key2]
#         features[count, key2, :, :] = p[0:34,0:26]
#     count = count + 1

labels = np.asarray(labels)

data2=zip(features,labels)


labelstest = np.asarray(labelstest)

data3=zip(featurestest,labelstest)

#name = 'norm_all'
name = 'train_norm_balanced_all2'
savePath = 'F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/dicts/datastage2_'+ name + '.p'

#The output file needs to be opened in binary mode:In Python 3, Binary modes 'wb', 'rb' must be specified whereas in Python 2x, they are not needed
f=open(savePath,'wb')
pickle.dump(data2,f)
f.close()

name = 'test_norm_balanced_all'
savePath = 'F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/dicts/datastage2_'+ name + '.p'

#The output file needs to be opened in binary mode:In Python 3, Binary modes 'wb', 'rb' must be specified whereas in Python 2x, they are not needed
f=open(savePath,'wb')
pickle.dump(data3,f)
f.close()


#Make tensor data
#dataset = tf.data.Dataset.from_tensor_slices((features,labels))



