
import _pickle as pickle
import numpy as np
import tensorflow as tf


f=open('F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/dicts/data_all.p','rb')
data=pickle.load(f)
f.close()

data = list(data)


feature_label =[]
features={}
labels=[]
count = 0


#34 X 26 is the common image size, some are 42 X 37 (need to crop central part for such cases- change this)
features = np.zeros((len(data),29,34,26))
for i in range(len(data)):
    feat_dict = data[i][0]
    labels.append(data[i][1])
    for key2 in feat_dict.keys():
        p = feat_dict[key2]
        features[count, key2, :, :] = p[0:34,0:26]
    count = count + 1

labels = np.asarray(labels)

data2=zip(features,labels)

name = 'all'
savePath = 'F:/UPENNACADS/CISBE537/PROJECT/B3537_2018/B3537_2018/dicts/datastage2_'+ name + '.p'

#The output file needs to be opened in binary mode:In Python 3, Binary modes 'wb', 'rb' must be specified whereas in Python 2x, they are not needed
f=open(savePath,'wb')
pickle.dump(data2,f)
f.close()



#Make tensor data
#dataset = tf.data.Dataset.from_tensor_slices((features,labels))



