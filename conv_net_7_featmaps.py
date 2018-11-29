
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 20:57:15 2016

@author: aoustimov
"""

import os
import sys
import timeit
import cPickle
import numpy as np

import theano
import theano.tensor as T

os.chdir('z:/Andrew/deep_learning/modules')


from logistic_sgd_2 import LogisticRegression, load_data_6
from mlp_2 import HiddenLayer  
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv



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
        
        
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (36-5+1 , 30-5+1) = (32, 26)
    # maxpooling reduces this further to (32/2, 26/2) = (16, 13)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 16, 13)
            
        layer1=LeNetConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],17,14),
        filter_shape=(nkerns[1],nkerns[0],2,3),
        poolsize=(2,2))
                    
        layer1_5=LeNetConvPoolLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],8,6),
        filter_shape=(nkerns[2],nkerns[1],3,1),
        poolsize=(2,2))



    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (50, 50 * 4 * 4) = (50, 800) with the default values.
        layer2_input = layer1_5.output.flatten(2)

    # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[2] * 3*3,
            n_out=5,
            activation=T.tanh
        )
        
    # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=5, n_out=2)
        
        self.p_y_given_x=layer3.p_y_given_x
        self.y_pred=layer3.y_pred
        self.negative_log_likelihood = layer3.negative_log_likelihood
        self.errors = layer3.errors
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.input = input

def evaluate_lenet5(learning_rate=0.02, n_epochs=200,
                    dataset='z:/Andrew/deep_learning/data/ppg_train_valid_test_36x30_feat_3_normed_overallsamples_2_4fold.p',
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

    datasets = load_data_6(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    print train_set_x.get_value().shape
    print valid_set_x.get_value().shape
    print test_set_x.get_value().shape
    print train_set_x.eval()[0]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    #declare the learning rate as a theano shared variable for decay in the 
    #gradient descent
    lr = theano.shared(0.02)    
    # start-snippet-1
    x = T.tensor4('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    classifier=conv_classifier(
                rng=rng,
                input=x,
                batch_size=batch_size,
                nkerns=nkerns)
    
    # the cost we minimize during training is the NLL of the model
    cost = classifier.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of gradients for all model parameters
    grads = [T.grad(cost, param) for param in classifier.params]
    
        
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - lr * grad_i)
        for param_i, grad_i in zip(classifier.params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 200  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
                #decaying the learning rate                
                lr = lr*0.99
                print 'learning rate @ iter ',iter,' = ', lr.eval()
            #right below is where the training functin is being implemented
            #it uses the update rule defined above, and the update asks for 
            #the gradient to be computed and multiplied by the learining rate
            cost_ij = train_model(minibatch_index)
            print 'cost ' ,cost_ij
                     

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                           
                    f = file('best_conv_ppg_7_featmaps_normedoversample_2_shuffled_cv_1.save','wb')
                    cPickle.dump(classifier.params,f,protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()
                    

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()
   
