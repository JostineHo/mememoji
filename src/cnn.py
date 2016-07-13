from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from keras.utils.np_utils import to_categorical
from log import save_model, save_config, save_result
from fer2013data import load_data
import numpy as np
import sys

X_train, y_train = load_data(sample_size=35887,
                             usage='Training',
                             labels=[0, 3]) # 0: angry, 3: happy

X_train = X_train.astype('float32')

a = X_train[y_train == 0][:10]
b = X_train[y_train == 1][:10]
X_train = np.vstack((a,b))
print X_train
c = y_train[y_train == 0][:10]
d = y_train[y_train == 1][:10]
y_train = np.hstack((c,d))
print y_train
y_train = to_categorical(y_train)

# params:
batch_size = 1
nb_epoch = 10

img_width, img_height =  X_train.shape[2:]
print 'X_train shape: ', X_train.shape # (n_sample, 1, 48, 48)
print 'y_train shape: ', y_train.shape # (n_sample, n_categories)
print '  img size: ', img_width, img_height
print 'batch size: ', batch_size
print '  nb_epoch: ', nb_epoch

# model architecture:
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                        input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# optimizer:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print 'Training....'
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
          validation_split=0.1, shuffle=True, verbose=1)

# set callback: https://github.com/sallamander/headline-generation/blob/master/headline_generation/model/model.py

loss_and_metrics = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
print 'Done!'
print 'Loss: ', loss_and_metrics[0]
print ' Acc: ', loss_and_metrics[1]

# logging:
notes = 'small set 20'
save_model(model.to_json(), '../data/results/')
save_config(model.get_config(), '../data/results/')
save_result(loss_and_metrics, notes, '../data/results/')
