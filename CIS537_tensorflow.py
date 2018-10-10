# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 13:11:15 2018

@author: Nehal Doiphode, Tse-Lun Hsu, Michael Zietz
"""

import os
import sys
import timeit
import _pickle as cPickle
import numpy as np
import tensorflow as tf
# import yaml


class LeNetConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
        
        assert image_shape[1] == filter_shape[1]
 
        self.input=input
        fan_in=np.prod(filter_shape[1:])
        fan_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        W_bound=np.sqrt(6./(fan_in + fan_out))
        self.W= theano.shared(
            np.asarray(rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                       dtype=theano.config.floatX),borrow=True)
        
        b_values=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,borrow=True)
        
        #convolve input feature maps with filters            
        conv_out=conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)

        #downsample each feature map individually via max pooling        
        pooled_out=downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True)
            
        self.output=T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]
        self.input = input               



if __name__ == '__main__':
	print ("hi")
    # with open("environment.yml", 'r') as stream:
	   #  try:
	   #      print(yaml.load(stream))
	   #  except yaml.YAMLError as exc:
	   #      print(exc)