from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# For displaying

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mimage

gs = gridspec.GridSpec(9, 9, top=1., bottom=0., right=1., left=0., hspace=0.,
        wspace=0.)


# Initialization
batch_size =32
nb_classes = 10
nb_epoch = 200
show_images = True
img_rows,img_cols = 32,32
img_channels = 3

# Load the dataset
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

print 'X_train shape: %s'%str( X_train.shape)
print 'X_test shape: %s' %str(X_test.shape)

count = 0
if show_images:
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(X_train[count].transpose((1, 2, 0)))
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

# Make target labels suitable for multi-class classification
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

#Define model

model = Sequential()


model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
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
# Start training
model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_test, y_test), shuffle=True)
# Saving wieghts
model.save_weights("model_cifar10.h5")
print("Saved model to disk")
