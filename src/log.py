from keras.models import  Sequential
import cPickle as pickle
import json
import time

localtime = time.asctime(time.localtime(time.time()))

def save_model(json_string, dirpath='../data/results/'):
    with open(dirpath + localtime +'.txt', 'w') as f:
        f.write(json_string)

def save_config(config, dirpath='../data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(localtime + '\n')
        f.write(str(config) + '\n')


def save_result(loss_and_metrics, notes='', dirpath='../data/results/'):
    with open(dirpath + 'result_log.txt', 'a') as f:
        f.write(localtime + 'comment: ' + notes + '\n' )
        f.write('Loss: ' + str(loss_and_metrics[0]) +
                ' Acc: ' + str(loss_and_metrics[1]) + '\n')
