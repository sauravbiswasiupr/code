#!/usr/bin/python 
'''Script to read the mnist data and explore each image pixel value '''
from PIL import Image 
from  matplotlib import pyplot as plt 
import numpy 
import cPickle 
from pylab import *

if __name__ =="__main__":
   filename='../data/mnist.pkl'
   f = open(filename,'rb')
   train_set , valid_set , test_set = cPickle.load(f) 
   print "train_set shape " , train_set[0].shape 
   print "Taking a sample image and exploring it ... "  
   im = train_set[0][1]
   im = numpy.reshape(im,(28,28))
   print "Image matrix values : "  , im 
   figure() 
   plt.imshow(im,interpolation='nearest')
   show()



   
