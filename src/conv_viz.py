from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import numpy as np
import sys

X_fname = '../data/X_train6_5pct.npy'
y_fname = '../data/y_train6_5pct.npy'
X_train = np.load(X_fname)
y_train = np.load(y_fname)
print 'Loading data...'
img_width, img_height = 48, 48

# build the VGG16 network
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='conv1_1', input_shape=(1, 48, 48)))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='conv1_2'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='conv1_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv2_1'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv2_2'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv2_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv3_1'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv3_2'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv3_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# set callback:
callbacks = []

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callbacks.append(early_stopping)

print 'Training....'
model.fit(X_train, y_train, nb_epoch=3, batch_size=256,
          validation_split=0.2, callbacks=callbacks, shuffle=True, verbose=1)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
model.save_weights('convnet_basic.h5', overwrite=True)

import h5py

f = model.load_weights('convnet_basic.h5')
for k in range(f['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break

    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')
