from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from keras.utils.np_utils import to_categorical
from fer2013data import load_data
import numpy as np
import sys

X_train = np.random.random((200, 1, 48, 48))
y_train = to_categorical(np.random.randint(0, 3, size=(200,)))
img_width, img_height =  X_train.shape[2:]
batch_size = 200
nb_epoch = 10

print 'X_train shape: ', X_train.shape # (2000, 1, 48, 48)
print 'y_train shape: ', y_train.shape # (2000, 1)
print '  img size: ', img_width, img_height
print 'batch size: ', batch_size
print '  nb_epoch: ', nb_epoch

'''Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
'''
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(3))
model.add(Activation('sigmoid'))

# optimizer:
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print 'Training....'
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
          validation_split=0.1, shuffle=True, verbose=1)

loss_and_metrics = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
print loss_and_metrics
