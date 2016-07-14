from keras.models import  Sequential
import cPickle as pickle
import json
import time

starttime = time.asctime(time.localtime(time.time()))

def save_model(json_string, dirpath='../data/results/'):
    with open(dirpath + starttime +'.txt', 'w') as f:
        f.write(json_string)

def save_config(config, dirpath='../data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(starttime + '\n')
        f.write(str(config) + '\n')

def save_result(train_val_accuracy, notes, conv_arch, dense,                            dirpath='../data/results/'):
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    with open(dirpath + starttime +'_train_val.txt', 'w') as f:
            f.write(str(train_acc) + '\n')
            f.write(str(val_acc) + '\n')

    endtime = time.asctime(time.localtime(time.time()))
    with open(dirpath + 'result_log.txt', 'a') as f:
        f.write(starttime + '--' + endtime + ' comment: ' + notes + '\n' )
        f.write(str(conv_arch) + ','+ str(dense) + '\n')
        f.write('Train acc: ' + str(train_acc[-1]) +
                'Val acc: ' + str(val_acc[-1]) +
                'Ratio: ' + str(val_acc[-1]/train_acc[-1]) + '\n')
