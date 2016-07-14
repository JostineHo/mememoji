from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from log import save_model, save_config, save_result
import numpy as np
import sys


def describe(X_train, y_train, batch_size, nb_epoch, conv_arch, dense):
    # training description:
    print 'emotion-counts: ', "[('Angry', 3995), ('Fear', 4097), ('Happy', 7215), ('Sad', 4830), ('Surprise', 3171), ('Neutral', 4965)]"
    print ' X_train shape: ', X_train.shape # (n_sample, 1, 48, 48)
    print ' y_train shape: ', y_train.shape # (n_sample, n_categories)
    print '      img size: ', X_train.shape[2], X_train.shape[3]
    print '    batch size: ', batch_size
    print '      nb_epoch: ', nb_epoch
    print 'conv architect: ', conv_arch
    print 'neural network: ', dense

def logging(model, notes, conv_arch, dense, train_val_accuracy, log=True):
    # model logging:
    save_model(model.to_json(), '../data/results/')
    save_config(model.get_config(), '../data/results/')
    save_result(train_val_accuracy, notes, conv_arch, dense,'../data/results/')

def cnn_architecture(X_train, y_train, notes, conv_arch=[3,3,3], dense=[2,64],
                                              batch_size=128, nb_epoch=100):

    X_train = X_train.astype('float32')
    describe(X_train, y_train, batch_size, nb_epoch, conv_arch, dense)

    # model architecture:
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',input_shape=(1, X_train.shape[2], X_train.shape[3])))

    if (conv_arch[0]-1) != 0:
        for i in range(conv_arch[0]-1):
            model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1] != 0:
        for i in range(conv_arch[1]):
            model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2] != 0:
        for i in range(conv_arch[2]):
            model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    if dense[0] != 0:
        for i in range(len(dense[0])):
            model.add(Dense(dense[1], activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print 'Training....'
    hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_split=0.3, shuffle=True, verbose=1)
    train_val_accuracy = hist.history
    # set callback: https://github.com/sallamander/headline-generation/blob/master/headline_generation/model/model.py

    # model result:
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print '          Done!'
    print '     Train acc: ', train_acc[-1]
    print 'Validation acc: ', val_acc[-1]
    print ' Overfit ratio: ', val_acc[-1]/train_acc[-1]
    logging(model, notes, conv_arch, dense, train_val_accuracy, log=True)
    return model

if __name__ == '__main__':
    # import dataset:
    notes = 'multicat 6 training 100% (batch_size=256, nb_epoch=100)'
    X_fname = '../data/X_train6.npy'
    y_fname = '../data/y_train6.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print 'Loading data...'

    cnn_architecture(X_train, y_train, notes, conv_arch=[3,2,1], dense=[1,64],
                                       batch_size=256, nb_epoch=100)
