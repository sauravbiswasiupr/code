"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import csv

import numpy
import datetime
import platform
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from random import choice
import itertools

DATASET = 'sicurambb.differencechannel.whitened.data'
NCLASSES = 2  # 2 for SICURA, 10 for MNIST
IMAGE_SIZE = 64
EPOCHS = 10
SERVER_NAME = platform.node()


class Parameter():
    def __init__(self, imagesize, fs1, ps1, fs2, ps2, fs3, 
                 n_epochs, batch_size, learning_rate, learning_rate_factor, nkerns, 
                 nhidden, n_out):
        self.imagesize = imagesize
        self.fs1 = fs1
        self.ps1 = ps1
        self.fs2 = fs2 
        self.ps2 = ps2
        self.fs3 = fs3
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_factor = learning_rate_factor
        self.nkerns = nkerns
        self.nhidden = nhidden
        self.n_out = n_out

    def __str__(self):
        return str((self.imagesize, 
                   self.fs1, 
                   self.ps1, 
                   self.fs2, 
                   self.ps2, 
                   self.fs3,
                   self.n_epochs,
                   self.batch_size,
                   self.learning_rate,
                   self.learning_rate_factor,
                   self.nkerns,
                   self.nhidden,
                   self.n_out))

    def values(self):
        return (self.imagesize, 
                self.fs1,
                self.ps1, 
                self.fs2, 
                self.ps2, 
                self.fs3,
                self.n_epochs,
                self.batch_size,
                self.learning_rate,
                self.learning_rate_factor,
                self.nkerns[0],
                self.nkerns[1],
                self.nhidden,
                self.n_out)



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weights to temporary values until we know the
        # shape of the output feature maps
        W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # replace weight values with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W.set_value(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]




def evaluate_sicura(parameter, dataset, verbose=True):
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

    isize = parameter.imagesize 
    learning_rate = parameter.learning_rate
    learning_rate_factor = parameter.learning_rate_factor
    n_epochs = parameter.n_epochs
    nkerns = parameter.nkerns
    batch_size = parameter.batch_size
    nhidden = parameter.nhidden
    n_out = parameter.n_out

    rng = numpy.random.RandomState(23455)

    if verbose: print 'started reading sicura data'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    if (n_train_batches <= 1) or (n_valid_batches <= 1) or (n_test_batches <= 1):
        raise Exception('n_train_batches == ' + str(n_train_batches) + 
                        '  n_valid_batches == ' + str(n_valid_batches) + 
                        '  n_test_batches == ' + str(n_test_batches) +
                        ' => should all >=2')


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    if verbose: print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, isize, isize))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, isize, isize),
            filter_shape=(nkerns[0], 1, parameter.fs1, parameter.fs1), 
            poolsize=(parameter.ps1, parameter.ps1))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)

    size2 = (isize - parameter.fs1 + 1) / parameter.ps1
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], size2, size2),
            filter_shape=(nkerns[1], nkerns[0], parameter.fs2, parameter.fs2), poolsize=(parameter.ps2, parameter.ps2))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * parameter.fs3 * parameter.fs3,
                         n_out=nhidden, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=nhidden, n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates dictionary by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = {}
    for param_i, grad_i in zip(params, grads):
        updates[param_i] = param_i - learning_rate * grad_i

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    if verbose: print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = epoch * n_train_batches + minibatch_index

            if iter % 100 == 0:
                if verbose: print 'training @ iter = ', iter
            train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:  print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                                  (epoch, minibatch_index + 1, n_train_batches, \
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
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    if verbose:  print(('     epoch %i, minibatch %i/%i, test error of best '
                                        'model %f %%') %
                                        (epoch, minibatch_index + 1, n_train_batches,
                                        test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    if verbose: print('Optimization complete.')
    if verbose: print('Best validation score of %f %% obtained at iteration %i,'\
                        'with test performance %f %%' %
                        (best_validation_loss * 100., best_iter, test_score * 100.))
    if verbose: print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    duration = ((end_time - start_time) / 60.0)
    return best_validation_loss * 100., best_iter, test_score * 100., duration, epoch


def init_logfile():
    logf_name = 'results.' + SERVER_NAME + '.csv'
    with open(logf_name, 'a') as f:
        logger = csv.writer(f)
        logger.writerow(('timestamp', 'size', 'fs1', 'ps1', 'fs2', 'ps2', 
                        'f3', 'n_epochs', 'batch_size', 'learning_rate', 
                        'learning_rate_factor', 'nkerns[0]', 'nkerns[1]',
                        'nhidden', 'n_out', 'best_validation_loss', 'best_iter',
                        'test_score', 'duration', 'actual_epochs', 'tag', 'server',
                        'datasetname'))
        f.close()
    return logf_name


def log_results(parameter, results, logf_name, tag='', datasetname=''):
    with open(logf_name, 'a') as f:
          logger = csv.writer(f)  
          logger.writerow((tuple([str(datetime.datetime.now())]) 
                          + parameter.values() + results + tuple([tag, SERVER_NAME, datasetname])))
          f.close()


def find_filter_and_pool_sizes(IMAGE_SIZE):
    valid_comb = []
    fs1_list = range(5, 13)
    ps1_list = range(2, 5)
    fs2_list = range(3, 10)
    ps2_list = range(2, 3)
    fs3_list = range(2, 8)
    for fs1, ps1, fs2, ps2, fs3 in list(itertools.product(fs1_list, ps1_list, fs2_list, ps2_list, fs3_list)):
        if ((float(IMAGE_SIZE) - fs1 + 1) / ps1 - fs2 + 1) / ps2 == fs3 and (fs1 > fs2): 
            valid_comb.append([fs1, ps1, fs2, ps2, fs3])
    return valid_comb


def calculate_nkerns(nkern1, imagesize, fs1, ps1, fs2):
    isize1 = imagesize - fs1 + 1
    isize2 = (isize1 / ps1) - fs2 + 1
    nkern2 = nkern1 * isize1 / isize2 
    return [nkern1, nkern2]


if __name__ == '__main__':
    logf_name = init_logfile()
    valid_comb = find_filter_and_pool_sizes(IMAGE_SIZE)
    i = 1   
    while True:
        
        # select values
        filter_and_pool_sizes = choice(valid_comb)
        fs1, ps1, fs2, ps2, fs3 = filter_and_pool_sizes
        learning_rate = choice([0.1, 0.01, 1, 10])
        batch_size = choice([50, 100, 10, 200])
        nhidden = choice([500, 100, 20, 200, 50])
        nkerns = calculate_nkerns(5, IMAGE_SIZE, fs1, ps1, fs2)

        best_test_error = 100
        best_parameter = None
        #for iteration, learning_rate_factor in enumerate([1, 2, 10]):
        for iteration, learning_rate_factor in enumerate([1]):
            tag = 'normal'
            parameter = Parameter(IMAGE_SIZE, fs1, ps1, fs2, ps2, fs3, 
                                n_epochs=EPOCHS, batch_size=batch_size, 
                                learning_rate=learning_rate, 
                                learning_rate_factor=learning_rate_factor,
                                nkerns=nkerns, nhidden=nhidden, n_out=NCLASSES)
            print i, parameter, tag, ; sys.stdout.flush()
            results = evaluate_sicura(parameter, DATASET, verbose=False)
            if results[2] < best_test_error:
                best_test_error = results[2]
                best_parameter = parameter
            log_results(parameter, results, logf_name, tag, DATASET)
            print str(results) 
            i += 1
        
        if best_test_error > 40.0:
            # dont compute further results if the first one is too bad
            # based on expericence that it does not get better
            continue

        # with different number of nkers
        for n_filter1 in [3, 10, 20]:
            tag = 'nkerns'
            parameter = best_parameter
            parameter.nkerns = calculate_nkerns(5, IMAGE_SIZE, fs1, ps1, fs2)
            print i, parameter, tag, ; sys.stdout.flush()
            results = evaluate_sicura(best_parameter, DATASET, verbose=False)
            if results[2] < best_test_error:
                best_test_error = results[2]
                best_parameter = parameter
            log_results(parameter, results, logf_name, tag, DATASET)
            print str(results)
            i += 1

        if results[2] < 18.0:  # how does increasing the n_epochs improve good runs?
            for i, epochs in enumerate([20, 50]):
                tag = 'inc_epochs'
                print i, parameter, tag, ; sys.stdout.flush()
                parameter.n_epochs = epochs
                results = evaluate_sicura(parameter, DATASET, verbose=False)
                log_results(parameter, results, logf_name, tag, DATASET)
                print str(results)
                i += 1               