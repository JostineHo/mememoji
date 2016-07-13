import pandas as pd
import numpy as np

def reconstruct(pix_str, size=(48,48)):
    pix_arr = np.array(map(int, pix_str.split()))
    return pix_arr.reshape(size)

def subset_data(df, usage='Training', labels=[0,3]):
    train = df[df.Usage == usage][['emotion','pixels']]
    train['pixels'] = train.pixels.apply(lambda x: reconstruct(x))
    subset = train[(train['emotion'] == labels[0]) | (train['emotion'] == labels[1])]
    y_train = subset.emotion
    lst = [mat for mat in subset.pixels]
    x = np.array(lst) # (n_samples, img_width, img_height)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    return X_train, y_train.values

def load_data(sample_size=35887, usage='Training', labels =[0,3], filepath='../data/fer2013.csv'):
    df = pd.read_csv(filepath)
    df = df[:sample_size]
    X_train, y_train = subset_data(df, usage=usage, labels=labels)
    num = [0,1,2,3,4,5,6]
    emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, cat in enumerate(np.unique(y_train)):
        y_train[np.where(y_train == cat)] = i
        print emotion[i] + ' is ' + str(i)
    return X_train, y_train

if __name__ == '__main__':
    X_train, y_train = load_data(sample_size=5000)
    print len(y_train)
