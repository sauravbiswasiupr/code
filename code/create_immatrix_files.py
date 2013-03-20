#!/usr/bin/python 
'''Module to read in Images and perform mean normalization on them , reshape them to size 256*256 and finally store them in an image matrix immatrix and gzip them 
@author : Saurav '''

from PIL import Image 
import numpy 
from matplotlib import pyplot as plt 
import gzip 
import theano 
import theano.tensor as T
import h5py 


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


def read_file(filename):
    '''Function to read a filename containing image names and their labels '''
    f = open(filename,'r')
    images=[] 
    labels=[]  
    for line in f.readlines():
        words = line.rstrip('\n').split(" ")
        img = '/home/saurav/Desktop/SICURA/query_images/query/'+words[0]
        labels.append(int(words[1]))
        im = Image.open(img) 
        #convert to grayscale 
        im_gr = im.convert('L')
        #reshape the image to be of size 256*256
        im_gr_res=im_gr.resize((256,256))
        im_res_arr = numpy.array(im_gr_res)/255.   #normalize the image to have pixel values between 0 and 1 
        #now flatten the image and unroll into a 1D vector to store in the images list 
        images.append(im_res_arr.flatten())
        print "Image : "  , img 
        
    immatrix = numpy.array(images,'f')
    labels = numpy.array(labels) 
    print "immatrix and labels created successfully .." 
    return immatrix , labels 

def load_datafile(filename): 

    immatrix,labels = read_file(filename) 
    train_x =  immatrix[:4000]
    train_y = labels[:4000]

    valid_x = immatrix[4000:6000]
    valid_y = labels[4000:6000] 

    test_x = immatrix[6000:]
    test_y = labels[6000:] 

    result =  [(train_x,train_y),(valid_x , valid_y) , (test_x,test_y)]
    fname = 'immatrix_8004.h5' 
    f= h5py.File(fname,'w')
    f['train_x'] = train_x
    f['train_y'] = train_y 
    f['valid_x'] = valid_x 
    f['valid_y'] = valid_y 
    f['test_x'] = test_x 
    f['test_y'] = test_y 
    
    f.close() 
    print "File gzipped successfully " 


if __name__ == "__main__": 

    filename = "dataset-labels.txt"
    load_datafile(filename) 

 
      
