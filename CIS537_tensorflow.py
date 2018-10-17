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


class LeNetConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
        
        assert image_shape[1] == filter_shape[1]
 
        self.input=input
        fan_in=np.prod(filter_shape[1:])
        fan_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        W_bound=np.sqrt(6./(fan_in + fan_out))


        #TODO need to change these theano to tensorlow
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
            
        #change 1
        # change it from self.output=T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        # to the live below
        self.output=tf.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]
        self.input = input  


class conv_classifier(object):
	def __init__(self,rng,input,batch_size,nkerns=[5,3,3]):
         
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
        layer0_input=input.reshape((batch_size,29,36,30))
        
        layer0 = LeNetConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 29, 36, 30),
        filter_shape=(nkerns[0], 29, 3, 3),
        poolsize=(2, 2))



def evaluate_lenet5(learning_rate=0.02, n_epochs=200,
                    nkerns=[10,10,10], batch_size=20):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = np.random.RandomState(13455)

    #TODO change this to tensorflow
    x = T.tensor4('x')   # the data is presented as rasterized images

    classifier=conv_classifier(
                rng=rng,
                input=x,
                batch_size=batch_size,
                nkerns=nkerns) 


if __name__ == '__main__':
	evaluate_lenet5()