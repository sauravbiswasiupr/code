#!/usr/bin/python 
'''the script reads in a list of image filenames from a text file and loads the corresponding images , if required normalizes them , and then stacks each image as a vector to make an ndarray of image arrays where each image , is resized to IMSIZE*IMSIZE and then unrolled into a 1D vector and stacked . We then prepare the train and test set and ultimately pickle it and gzip it so that it can be later unpickled and used by any classifier in the format required '''

from PIL import Image 
import numpy 
import pylab 
import os
from scipy.misc import imread 
IMSIZE = 256   #let it be 256 for now , we can later change this to a user argument 
from numpy import zeros,ones,hstack,vstack
import theano 
import theano.tensor as T
from pickle import dump
import time
import random as pyrandom 

##imports done 

def create_im_list(filename):
   '''Function to read a filename and create a list of files contained in that file '''

   f = open(filename,'r')
   imlist=[]
   for line in f.readlines(): 
       words = line.rstrip('\n').split(" ")
       print words[0] , words[1]
       imlist.append((words[0],words[1])) 
       #note that each imlist entry is a tuple consisting of the image name and a label if it has a gun or not 
       print "error not there " 
   return imlist 



def create_image_array(filename): 
    '''Function that takes a filename as parameter and stacks a numpy array of images rolled into 1D vector from that filename list '''
    imlist = create_im_list(filename) #note that a list of tuples is returned 
    imnbr = len(imlist) 
    
    #change to the directory where query images are present 
    folder = '/home/saurav/Desktop/SICURA/query_images/query/'
    images =[]
    labels=[] 
    for i  in range(imnbr):
        print "Image :" , imlist[i][0]  
        im = Image.open(folder+imlist[i][0])
        labels.append(int(imlist[i][1]))
        print im 
        #convert image to grayscale 
        im = im.convert('L')
        im_resized = im.resize((256,256))
        im_resized = numpy.array(im_resized)/255.  #normalize the image to make pixels  values lie between 0 and 1 
        print " Resized image shape : "  , im_resized.shape
        #im_reshaped = numpy.reshape(im_resized,(64*64,))
       
        
        #now flatten (unroll) the image into a 1D array and then append to the images[] list which will be later converted into a numpy array 
        images.append(im_resized.flatten())
        #images.append(im_reshaped)


    #now convert the image list into a numpy array along with the labels 
    immatrix = numpy.array(images,'f')
    print "Immatrix shape : "  , immatrix.shape
    print "Mean normalizing the image matrix ...."  
  
    time.sleep(2)
    labels = numpy.array(labels)
    return immatrix , labels 

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

def load_datafile(filename):
    
    
    immatrix,labels = create_image_array(filename)
    print "The first 4 entries of the entire image matrix : " , immatrix[:4][:]
    print "Labels : "  , labels[:4]
    print labels.shape
    #we have 8000 images so we wll create a 0 ...8003 numbered list and randomize it and then create the immatrix and labels but this time randomized 
    indices = [i for i in range(8004)]
    immatrix_temp =[]
    labels_temp =[] 
    pyrandom.shuffle(indices) 
    for i in indices: 
        immatrix_temp.append(immatrix[i])
        labels_temp.append(labels[i])

    immatrix = numpy.array(immatrix_temp , 'f')
    labels = numpy.array(labels_temp)
    
    # print "Storing the image matrix and labels in savez format ... " 
    #numpy.savez("immatrix-labels.npz",immatrix , labels )
    #print "Matrix saved successfully "  
    #now time tom divide the data into the train , validation and test set , first 4000 included into training set , from 4000 to 6000 in validation set and the remaining are in 
    #
    train_x =  immatrix[:4000]
    train_y = labels[:4000]

    valid_x = immatrix[4000:6000]
    valid_y = labels[4000:6000] 

    test_x = immatrix[6000:]
    test_y = labels[6000:] 

    result =  [(train_x,train_y),(valid_x , valid_y) , (test_x,test_y)]
    print "Image dimensions "  , immatrix[0].shape
    #now pickle the result and dump into a file 
    # dataset ='sicura_old_8004.pkl' 
    train_set_x , train_set_y = shared_dataset(result[0])
    valid_set_x , valid_set_y = shared_dataset(result[1])
    test_set_x , test_set_y = shared_dataset(result[2])
    
    result = [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return result 



