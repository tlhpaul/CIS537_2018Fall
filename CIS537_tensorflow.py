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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

tf.logging.set_verbosity(tf.logging.INFO)

class conv_classifier(object):
	def __init__(feature, lables, mode, batch_size):
         
  		# Input Layer
  		# TODO check the size
  		input_layer = tf.reshape(features["x"], [batch_size, 29, 36, 30])

  		# Convolutional Layer #1
  		# Applies 32 5x5 filters (extracting 5x5-pixel subregions), with tanh activation function
  		# TODO check filter and kernel size
  		conv1 = tf.layers.conv2d(
      		inputs=input_layer,
      		filters=32,
      		kernel_size=[5, 5],
      		padding="same",
      		activation=tf.nn.tanh)

  		# Pooling Layer #1
  		# Performs max pooling with a 2x2 filter and stride of 2 
  		# (which specifies that pooled regions do not overlap)
  		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  		# Convolutional Layer #2 and Pooling Layer #2
  		# Applies 64 5x5 filters, with tanh activation function
  		conv2 = tf.layers.conv2d(
      		inputs=pool1,
      		filters=64,
      		kernel_size=[5, 5],
      		padding="same",
      		activation=tf.nn.tanh)

  		# Again, performs max pooling with a 2x2 filter and stride of 2
  		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  		# Dense Layer
  		pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  		
  		#1,024 neurons, with dropout regularization rate of 0.4 
  		# (probability of 0.4 that any given element will be 
  		# dropped during training)
  		dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.tanh)
  		dropout = tf.layers.dropout(
      		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  		# Logits Layer
  		# 10 neurons, one for each digit target class (0–9).
  		logits = tf.layers.dense(inputs=dropout, units=10)

  		predictions = {
      		# Generate predictions (for PREDICT and EVAL mode)
      		"classes": tf.argmax(input=logits, axis=1),
      		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      		# `logging_hook`.
      		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  		}

  		if mode == tf.estimator.ModeKeys.PREDICT:
    		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  		# Calculate Loss (for both TRAIN and EVAL modes)
  		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  		# Configure the Training Op (for TRAIN mode)
  		if mode == tf.estimator.ModeKeys.TRAIN:
    		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    		train_op = optimizer.minimize(
        		loss=loss,
        		global_step=tf.train.get_global_step())
    		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  		# Add evaluation metrics (for EVAL mode)
  		eval_metric_ops = {
    	  	"accuracy": tf.metrics.accuracy(
        		  labels=labels, predictions=predictions["classes"])}

  		return tf.estimator.EstimatorSpec(
      		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def evaluate_lenet5(dataset='z:/Andrew/deep_learning/data/ppg_train_valid_test_36x30_feat_3_normed_overallsamples_2_4fold.p',
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
    
    data = tf.contrib.learn.datasets.load_dataset(dataset)
  	train_data = data.train.images # Returns np.array
  	train_labels = np.asarray(data.train.labels, dtype=np.int32)
  	eval_data = data.test.images # Returns np.array
  	eval_labels = np.asarray(data.test.labels, dtype=np.int32)


  	# The model_dir argument specifies the directory 
  	# where model data (checkpoints) will be saved
    classifier = tf.estimator.Estimator(
    	model_fn=conv_classifier, model_dir="CIS537_tensorflow")

    # Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
    	tensors=tensors_to_log, every_n_iter=50)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    	x={"x": train_data},
    	y=train_labels,
    	batch_size=100,
    	num_epochs=None,
    	shuffle=True)

	classifier.train(
    	input_fn=train_input_fn,
    	steps=20000,
    	hooks=[logging_hook])


	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    	x={"x": eval_data},
    	y=eval_labels,
    	num_epochs=1,
    	shuffle=False)

	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)


if __name__ == '__main__':
	evaluate_lenet5()