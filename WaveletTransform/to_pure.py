import pandas as pd
import numpy as np
from pathlib import Path
import pickle

origin_dir = Path('./OriginData')
orgin_file = origin_dir / 'TX_price.csv'
df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
dates = df['date']
features = ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
time_steps = 16
test_size = 250
train_size = df.shape[0] - test_size


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
    train_data['y'] = df['adj_close'][time_steps:train_size+time_steps].values
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
    test_data['y'] = df['adj_close'][train_size + time_steps:].values
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

    labels = np.sign(np.diff(df['adj_close'].values)).astype(int)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size].values
    train_data['y'] = labels[time_steps-1:train_size-1]
    for feature in features:
        train_data[feature] = list()
        for i in range(train_size - time_steps):
            train_data[feature] .append(df[feature][i:i+time_steps].values)

    train_file = Path('./pure_{}/train_binary.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size:].values
    test_data['y'] = labels[train_size-1:]
    for feature in features:
        test_data[feature] = list()
        for i in range(test_size):
            test_data[feature].append(df[feature][i - time_steps + train_size: i + train_size].values)

    test_file = Path('./pure_{}/test_binary.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_multiclass():
    '''
    How to read the traing set and the testing set?
    with open('./{}_{}/train.pkl'.format(wavelet, time_steps), 'rb') as f:
        a = pickle.load(f)
    with open('./{}_{}/test.pkl'.format(wavelet, time_steps), 'rb') as f:
        b = pickle.load(f)

    lebel:
            0 -> long 做多
            1 -> short 做空
            2 -> nop_up 不動作，但股票漲
            3 -> nop_down 不動作，但股票跌
    '''

    _open, _close = df['adj_open'].values, df['adj_close'].values
    fee = (_open * 1.425 * 1e-3 + _close * 4.425 * 1e-3)
    long = (_close - _open) - fee > 0
    short = (_open - _close) - fee > 0
    nop_up = ~(long | short) & (_close > _open)
    nop_down = ~(long | short) & (_open > _close)
    acts = [long, short, nop_up, nop_down]

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = np.zeros(train_size, dtype=int)

    for i, act in enumerate(acts):
        train_data['y'][act[time_steps:train_size + time_steps]] = i

    for feature in features:
        train_data[feature] = list()
        for i in range(train_size):
            train_data[feature] .append(df[feature][i:i+time_steps].values)

    train_file = Path('./pure_{}/train_multiclass.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = np.zeros(test_size, dtype=int)

    for i, act in enumerate(acts):
        test_data['y'][act[train_size + time_steps:]] = i

    for feature in features:
        test_data[feature] = list()
        for i in range(train_size, train_size + test_size):
            test_data[feature].append(df[feature][i:i + time_steps].values)

    test_file = Path('./pure_{}/test_multiclass.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":

    # create training set and data set by day
    # create_dataset()
    # create_dataset_binary()
    create_dataset_multiclass()
