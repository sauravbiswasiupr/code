#!/usr/bin/python 
'''Program to view the data that Ludwig developed for his version of LeNet '''

from PIL import Image 
from matplotlib import pyplot as plt 
import numpy 
import cPickle 
from pylab import * 


def read_data(filename):
    result = cPickle.load(filename) 
    training_set , valid_set , test_set =  result 
    print "Training set shape : "  , training_set[0].shape 
    print "Displaying a sample image from the training set : "  
    im = training_set[0][0] 
    print "Image shape "  , im.shape 
    figure()
    imarray = numpy.reshape(im,(64,64))
    image = plt.imshow(imarray,interpolation='nearest')
    show()
    



