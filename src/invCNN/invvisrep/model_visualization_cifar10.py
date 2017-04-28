#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 21:54:39 2016

@author: avijit
"""

from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop
#from keras.utils import np_utils
import keras.backend as K
# For displaying
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.image as mimage

gs = gridspec.GridSpec(9, 9, top=1., bottom=0., right=1., left=0., hspace=0.,
        wspace=0.)


# Initialization
batch_size =32
nb_classes = 10
nb_epoch = 200
show_images = False
img_rows,img_cols = 32,32
img_channels = 3

# Load the dataset
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

print 'X_train shape: %s'%str( X_train.shape)
print 'X_test shape: %s' %str(X_test.shape)



if show_images:
    print "Sample images from Cifar-10 train set"
    count = 0
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(X_train[count].transpose((1, 2, 0)))
        count += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()
    

if show_images:
    print "Sample images from Cifar-10 test set"
    count = 0
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(X_test[count].transpose((1, 2, 0)))
        count += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()
# Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255




#Define model

model = Sequential()


model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy',
              optimizer = sgd,metrics = ['accuracy'])

#model.summary()

# Load weights
print "Loading weights for cifar-10 model.."
model.load_weights('model_cifar10.h5')
print "Done!"

# Define layer of interest
layer_of_interest = "convolution2d_2"
lnames=[l.name for l in model.layers]
layer_index = lnames.index(layer_of_interest) + 2 # See paper

# Get the output for all training images from the layer_of_interest


intermediate_layer_output_func = K.function([model.layers[0].input, K.learning_phase()],
                                        model.layers[layer_index].output)

intermediate_layer_output = np.zeros((X_train.shape[0], model.layers[layer_index].output_shape[1], 
                                     model.layers[layer_index].output_shape[2], model.layers[layer_index].output_shape[3]))

for nbatch in range(X_train.shape[0] / batch_size):
        sys.stdout.write("\rProcessing batch %s/%s" %
                         (nbatch + 1, len(range(X_train.shape[0] / batch_size))))
        sys.stdout.flush()
        X = X_train[nbatch * batch_size: (nbatch + 1) * batch_size]
        intermediate_layer_output[nbatch * batch_size: (nbatch + 1) * batch_size] = intermediate_layer_output_func([X, 0])
#
#
#if show_images:
#    print "Sample images from Cifar-10 train set"
#    count = 0
#    for g in gs:
#        ax = plt.subplot(g)
#        ax.imshow(intermediate_layer_output[count,60])
#        count += 1
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ax.set_aspect('auto')
#    plt.show()

#print intermediate_layer_output.shape


decoder_model = Sequential()

decoder_model.add(Convolution2D(64, 3, 3, subsample = (1,1),  border_mode='same',
                        input_shape=intermediate_layer_output.shape[1:]))
decoder_model.add(Activation('relu'))
decoder_model.add(Convolution2D(64, 3, 3, subsample = (1,1),  border_mode='same'))
decoder_model.add(Activation('relu'))
decoder_model.add(Convolution2D(64, 3, 3, subsample = (1,1),  border_mode='same'))
decoder_model.add(Activation('relu'))


# decoder_model.add(ZeroPadding2D(padding=(1,1)))
#decoder_model.add(Deconvolution2D(64, 5, 5, output_shape=(None, 64, 8, 8), subsample = (2,2), border_mode = 'same'))
#decoder_model.add(Activation('relu'))
#decoder_model.add(Deconvolution2D(32, 5, 5,output_shape=(None, 32, 16, 16), subsample = (2,2), border_mode = 'same'))
#decoder_model.add(Activation('relu'))
decoder_model.add(Deconvolution2D(3, 5, 5,output_shape=(None, 3, 32, 32), subsample = (2,2), border_mode = 'same'))
decoder_model.add(Activation('relu'))
#decoder_model.add(Deconvolution2D(8, 5, 5,output_shape=(None, 8, 96, 96), subsample = (2,2), border_mode = 'same'))
#decoder_model.add(Activation('relu'))
#decoder_model.add(Deconvolution2D(3, 5, 5, output_shape=(None, 3, 32,32), subsample = (2,2), border_mode = 'same'))
#decoder_model.add(Activation('relu'))
#decoder_model.add(Deconvolution2D(256, 5, 5,output_shape=(None, 3, 25, 25),  subsample = (2,2), border_mode = 'same'))


# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
rms= RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
decoder_model.compile(loss = 'mean_squared_error',
              optimizer = rms,metrics = ['mean_absolute_percentage_error'])

#print decoder_model.layers[-1].output_shape

decoder_model.summary()
# Start training
decoder_model.fit(intermediate_layer_output, X_train, batch_size = batch_size,validation_split = 0.2, nb_epoch = 30, shuffle=True)
# Saving wieghts
decoder_model.save_weights("decoder_model_cifar10_.h5")
print("Saved model to disk")



decoder_model.load_weights('decoder_model_cifar10_.h5')


intermediate_layer_output_test = np.zeros((X_test.shape[0], model.layers[layer_index].output_shape[1], 
                                     model.layers[layer_index].output_shape[2], model.layers[layer_index].output_shape[3]))

for nbatch in range(X_test.shape[0] / batch_size):
        sys.stdout.write("\rProcessing batch %s/%s" %
                         (nbatch + 1, len(range(X_test.shape[0] / batch_size))))
        sys.stdout.flush()
        X = X_test[nbatch * batch_size: (nbatch + 1) * batch_size]
        intermediate_layer_output_test[nbatch * batch_size: (nbatch + 1) * batch_size] = intermediate_layer_output_func([X, 0])
        
        

output = np.zeros(X_test.shape)

output = decoder_model.predict(intermediate_layer_output_test)
#
#
np.save("output.npy", output)














