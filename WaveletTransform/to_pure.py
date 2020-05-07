import pandas as pd
import numpy as np
from pathlib import Path
import pickle

origin_dir = Path('./OriginData')
orgin_file = origin_dir / 'TX_price.csv'
df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
dates = df['date']
features = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
time_steps = 16
train_size = int(df.shape[0] * 0.94)
test_size = df.shape[0] - train_size - time_steps


def create_dataset():
    '''
    How to read the traing set and the testing set?
    with open('./{}_{}/train.pkl'.format(wavelet, time_steps), 'rb') as f:
        a = pickle.load(f)
    with open('./{}_{}/test.pkl'.format(wavelet, time_steps), 'rb') as f:
        b = pickle.load(f)
    '''

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = df['close'][time_steps:train_size+time_steps].values
    for feature in features:
        train_data[feature] = list()
        for i in range(train_size):
            train_data[feature] .append(df[feature][i:i+time_steps].values)

    train_file = Path('./pure_{}/train.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['close'][train_size + time_steps:].values
    for feature in features:
        test_data[feature] = list()
        for i in range(train_size, train_size + test_size):
            test_data[feature].append(df[feature][i:i + time_steps].values)

    test_file = Path('./pure_{}/test.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_binary():
    '''
    How to read the traing set and the testing set?
    with open('./{}_{}/train.pkl'.format(wavelet, time_steps), 'rb') as f:
        a = pickle.load(f)
    with open('./{}_{}/test.pkl'.format(wavelet, time_steps), 'rb') as f:
        b = pickle.load(f)
    '''

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = np.sign(np.diff(df['close'][time_steps-1:train_size + time_steps].values)).astype(int)
    for feature in features:
        train_data[feature] = list()
        for i in range(train_size):
            train_data[feature] .append(df[feature][i:i+time_steps].values)

    train_file = Path('./pure_{}/train_binary.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = np.sign(np.diff(df['close'][train_size-1 + time_steps:].values)).astype(int)
    for feature in features:
        test_data[feature] = list()
        for i in range(train_size, train_size + test_size):
            test_data[feature].append(df[feature][i:i + time_steps].values)

    test_file = Path('./pure_{}/test_binary.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":

    # create training set and data set by day
    create_dataset()
    create_dataset_binary()
