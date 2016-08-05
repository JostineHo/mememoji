from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, MaxoutDense
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from log import save_model, save_config, save_result
import numpy as np
import time
import sys

def describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, maxdense):
    print ' X_train shape: ', X_shape # (n_sample, 1, 48, 48)
    print ' y_train shape: ', y_shape # (n_sample, n_categories)
    print '      img size: ', X_shape[2], X_shape[3]
    print '    batch size: ', batch_size
    print '      nb_epoch: ', nb_epoch
    print '       dropout: ', dropout
    print 'conv architect: ', conv_arch
    print 'neural network: ', maxdense

def logging(model, starttime, batch_size, nb_epoch, conv_arch, maxdense, dropout,
            X_shape, y_shape, train_acc, val_acc, dirpath):
    now = time.ctime()
    model.save_weights('../data/weights/{}'.format(now))
    save_model(model.to_json(), now, dirpath)
    save_config(model.get_config(), now, dirpath)
    save_result(starttime, batch_size, nb_epoch, conv_arch,maxdense, dropout,
                    X_shape, y_shape, train_acc, val_acc, dirpath)

def cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)],
                    maxdense=[64,2], dropout=0.5, batch_size=128, nb_epoch=100, validation_split=0.2, patience=5, dirpath='../data/results/'):
    starttime = time.time()
    X_train = X_train.astype('float32')
    X_shape = X_train.shape
    y_shape = y_train.shape
    describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, maxdense)

    # model architecture:
    model = Sequential()
    model.add(Convolution2D(conv_arch[0][0], 3, 3, border_mode='same', activation='relu',input_shape=(1, X_train.shape[2], X_train.shape[3])))

    if (conv_arch[0][1]-1) != 0:
        for i in range(conv_arch[0][1]-1):
            model.add(Convolution2D(conv_arch[0][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            model.add(Convolution2D(conv_arch[1][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            model.add(Convolution2D(conv_arch[2][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    if maxdense[1] != 0:
        for i in range(maxdense[1]):
            model.add(MaxoutDense(maxdense[0], nb_feature=1))
            if dropout:
                model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # set callback:
    callbacks = []
    if patience != 0:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        callbacks.append(early_stopping)

    print 'Training....'
    hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_split=validation_split, callbacks=callbacks, shuffle=True, verbose=1)

    # model result:
    train_val_accuracy = hist.history
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print '          Done!'
    print '     Train acc: ', train_acc[-1]
    print 'Validation acc: ', val_acc[-1]
    print ' Overfit ratio: ', val_acc[-1]/train_acc[-1]

    logging(model, starttime, batch_size, nb_epoch, conv_arch, maxdense, dropout, X_shape, y_shape, train_acc, val_acc, dirpath)

    return model

if __name__ == '__main__':
    # import testing dataset:
    X_fname = '../data/X_train6_5pct.npy'
    y_fname = '../data/y_train6_5pct.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print 'Loading data...'

    cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)], maxdense=[64,2], batch_size=256, nb_epoch=5, dirpath = '../data/results/')
