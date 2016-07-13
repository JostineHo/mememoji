from collections import defaultdict, Counter
from matplotlib import pyplot
from shutil import copyfile
import cPickle as pickle
import numpy as np
import os

def collect_emotion_labels():
    emotion_labels = defaultdict(int)
    for root, dirs, files in os.walk(".", topdown=False):
        if 'Emotion' in root:
            for fname in files:
                if fname != '.DS_Store':
                    with open(root + '/' + fname,'r') as f:
                        value = f.readline().strip().split('.')
                        emotion_labels[fname.strip('_emotion.txt')] = int(value[0])
    return emotion_labels

def summary(emotion_labels):
    labels = np.array(emotion_labels.values())
    return Counter(labels)

def gather_labeled_images(emotion_labels):
    ck_image_paths = ['Cohn-Kanade Images/' + fname.split('_')[0] +
                                        '/' + fname.split('_')[1] +
                                        '/' + fname+'.png' for fname in emotion_labels]
    for path in ck_image_paths:
        fname = path.split('/')[-1]
        copyfile('data/'+ path,
                 'data/LabeledImages/' + str(emotion_labels[fname.strip('.png')]) + '/' + fname)

if __name__ == '__main__':
    emotion_labels = collect_emotion_labels()
    gather_labeled_images(emotion_labels)
    print summary(emotion_labels)
