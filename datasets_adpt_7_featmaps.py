# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:12:23 2016

@author: OustimoA
"""

import cPickle
import numpy as np
import random


f=open('Z:../deep_learning/data_combo_3_normed_overallsamples.p','r')
data=cPickle.load(f)
f.close()

type(data)
len(data)
len(data[0])
len(data[0][0])
len(data[0][1])
data[0][1]
#split into case/control and into feature/target

seed=2323

controls=[]
cases=[]
for i in np.arange(len(data)):
    if data[i][1]==0:
        controls.append(data[i])
    else:
        cases.append(data[i])
    
controls_feat_list=[]
controls_target_list=[]
for feat_dict, target in controls:
    controls_target_list.append(target)
    each_feat_list=[]
    for key, item in feat_dict.iteritems():
        each_feat_list.append(item)
    controls_feat_list.append(each_feat_list)

cases_feat_list=[]
cases_target_list=[]
for feat_dict, target in cases:
    cases_target_list.append(target)
    each_feat_list=[]
    for key, item in feat_dict.iteritems():
        each_feat_list.append(item)
    cases_feat_list.append(each_feat_list)

#for controls
#making train, validation, and test sets
#making a list of random non-repeating integers for indeces of control-train set
###############################################################################
random.seed(seed)

cont_ix=[]
a = list(np.arange(len(controls_feat_list)))
for i in xrange(int(len(controls_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    cont_ix.append(b)
co_f1=[]
for ix in cont_ix:
    co_f1.append(controls_feat_list[ix])
co_f1_left=[]    
for ix in np.arange(len(controls_feat_list)):
    if ix not in cont_ix:
        co_f1_left.append(controls_feat_list[ix])

co_t1 = np.zeros(len(co_f1))

###############################################################################
cont_ix=[]
a = list(np.arange(len(co_f1_left)))
for i in xrange(int(len(controls_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    cont_ix.append(b)
co_f2=[]
for ix in cont_ix:
    co_f2.append(co_f1_left[ix])
co_f2_left=[]    
for ix in np.arange(len(co_f1_left)):
    if ix not in cont_ix:
        co_f2_left.append(co_f1_left[ix])

co_t2 = np.zeros(len(co_f2))
###############################################################################
cont_ix=[]
a = list(np.arange(len(co_f2_left)))
for i in xrange(int(len(controls_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    cont_ix.append(b)
co_f3=[]
for ix in cont_ix:
    co_f3.append(co_f2_left[ix])
co_f3_left=[]    
for ix in np.arange(len(co_f2_left)):
    if ix not in cont_ix:
        co_f3_left.append(co_f2_left[ix])

co_t3 = np.zeros(len(co_f3))
###############################################################################
co_f4 = co_f3_left
co_t4 = np.zeros(len(co_f4))
###############################################################################
###############################################################################
#split cases into four mutually exclusive groups
case_ix=[]
a = list(np.arange(len(cases_feat_list)))
for i in xrange(int(len(cases_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    case_ix.append(b)
ca_f1=[]
for ix in case_ix:
    ca_f1.append(cases_feat_list[ix])
ca_f1_left=[]    
for ix in np.arange(len(cases_feat_list)):
    if ix not in case_ix:
        ca_f1_left.append(cases_feat_list[ix])

ca_t1 = np.ones(len(ca_f1))
###############################################################################
case_ix=[]
a = list(np.arange(len(ca_f1_left)))
for i in xrange(int(len(cases_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    case_ix.append(b)
ca_f2=[]
for ix in case_ix:
    ca_f2.append(ca_f1_left[ix])
ca_f2_left=[]    
for ix in np.arange(len(ca_f1_left)):
    if ix not in case_ix:
        ca_f2_left.append(ca_f1_left[ix])

ca_t2 = np.ones(len(ca_f2))
###############################################################################
case_ix=[]
a = list(np.arange(len(ca_f2_left)))
for i in xrange(int(len(cases_feat_list)*0.25)):
    b = a[random.randint(0,len(a)-i)]
    a.remove(b)
    case_ix.append(b)
ca_f3=[]
for ix in case_ix:
    ca_f3.append(ca_f2_left[ix])
ca_f3_left=[]    
for ix in np.arange(len(ca_f2_left)):
    if ix not in case_ix:
        ca_f3_left.append(ca_f2_left[ix])

ca_t3 = np.ones(len(ca_f3))
###############################################################################
ca_f4 = ca_f3_left
ca_t4 = np.ones(len(ca_f4))
###############################################################################
###############################################################################
train_f1 = co_f1+co_f2+ca_f1+ca_f2
train_f2 = co_f1+co_f4+ca_f1+ca_f4
train_f3 = co_f3+co_f4+ca_f3+ca_f4
train_f4 = co_f2+co_f3+ca_f2+ca_f3

train_t1 = list(co_t1)+list(co_t2)+list(ca_t1)+list(ca_t2)
train_t2 = list(co_t1)+list(co_t4)+list(ca_t1)+list(ca_t4)
train_t3 = list(co_t3)+list(co_t4)+list(ca_t3)+list(ca_t4)
train_t4 = list(co_t2)+list(co_t3)+list(ca_t2)+list(ca_t3)

valid_f1 = co_f3+ca_f3
valid_f2 = co_f2+ca_f2
valid_f3 = co_f1+ca_f1
valid_f4 = co_f4+ca_f4

valid_t1 = list(co_t3)+list(ca_t3)
valid_t2 = list(co_t2)+list(ca_t2)
valid_t3 = list(co_t1)+list(ca_t1)
valid_t4 = list(co_t4)+list(ca_t4)

test_f1 = co_f4+ca_f4
test_f2 = co_f3+ca_f3
test_f3 = co_f2+ca_f2
test_f4 = co_f1+ca_f1

test_t1 = list(co_t4)+list(ca_t4)
test_t2 = list(co_t3)+list(ca_t3)
test_t3 = list(co_t2)+list(ca_t2)
test_t4 = list(co_t1)+list(ca_t1)
###############################################################################
train_1_zip = zip(train_f1,train_t1)
train_2_zip = zip(train_f2,train_t2)
train_3_zip = zip(train_f3,train_t3)
train_4_zip = zip(train_f4,train_t4)

train_1_zip_copy = train_1_zip
train_2_zip_copy = train_2_zip
train_3_zip_copy = train_3_zip
train_4_zip_copy = train_4_zip

#shuffeling the training set so that the classes are shuffled and facilitate
#better SGD training... hopefully... :) 
np.random.shuffle(train_1_zip_copy)
np.random.shuffle(train_2_zip_copy)
np.random.shuffle(train_3_zip_copy)
np.random.shuffle(train_4_zip_copy)

train_1 = [zip(*train_1_zip_copy)[0],zip(*train_1_zip_copy)[1]]
train_2 = [zip(*train_2_zip_copy)[0],zip(*train_2_zip_copy)[1]]
train_3 = [zip(*train_3_zip_copy)[0],zip(*train_3_zip_copy)[1]]
train_4 = [zip(*train_4_zip_copy)[0],zip(*train_4_zip_copy)[1]]

#train_1 = [train_f1, train_t1]
#train_2 = [train_f2, train_t2]
#train_3 = [train_f3, train_t3]
#train_4 = [train_f4, train_t4]

valid_1 = [valid_f1, valid_t1]
valid_2 = [valid_f2, valid_t2]
valid_3 = [valid_f3, valid_t3]
valid_4 = [valid_f4, valid_t4]

test_1 = [test_f1, test_t1]
test_2 = [test_f2, test_t2]
test_3 = [test_f3, test_t3]
test_4 = [test_f4, test_t4]
###############################################################################
data_1 = tuple([train_1,valid_1,test_1])
data_2 = tuple([train_2,valid_2,test_2])
data_3 = tuple([train_3,valid_3,test_3])
data_4 = tuple([train_4,valid_4,test_4])
###############################################################################
return_data = tuple([data_1,data_2,data_3,data_4])
##############################################################################

f=open('z:../deep_learning/ppg_train_valid_test_36x30_feat_3_normed_overallsamples_2_4fold.p','w')
cPickle.dump(return_data,f)
f.close()

f=open('z:../deep_learning/ppg_train_valid_test_36x30_feat_3_normed_overallsamples_2_4fold.p','r')
data=cPickle.load(f)
f.close()

len(data[0])
len(data[0][0])
len(data[0][0][0])
len(data[0][0][0][0])
data[0][0][0][0][0].shape

rval=load_data_6('z:../deep_learning/ppg_train_valid_test_36x30_feat_3_normed_ineachsample_4fold.p')
len(rval)
len(rval[0])
len(rval[0][0].eval())
rval[0][0].eval()[0].shape

import os

# current working directory
# os.chdir('z:/Andrew/deep_learning/modules')

def load_data_6(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    f = open(dataset, 'r')
    train_set, valid_set, test_set = pickle.load(f)[0]
    f.close()
        
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        shared_x = tf.Variable(np.asarray(data_x))
        shared_y = tf.Variable(np.asarray(data_x))


        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


##############################################################################