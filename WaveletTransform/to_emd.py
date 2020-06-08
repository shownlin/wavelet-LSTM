import pandas as pd
from PyEMD import EMD
import numpy as np
from pathlib import Path
import pickle

origin_dir = Path('./OriginData')
orgin_file = origin_dir / 'TX_price.csv'
df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
dates = df['date']
features = ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume', 'buy_sell']
time_steps = 16
test_size = 250
train_size = df.shape[0] - test_size
max_imf = 3


def to_emd():
    '''
    How to read pickle file in python?
    with open('./EMD/open_imf_list.pkl', 'rb') as f:
        data = pickle.load(f)
    '''

    skip = list()
    for feature in features:
        save_file = Path('./EMD_{}/{}_imf_list.pkl'.format(time_steps, feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            if i in skip:
                continue
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            emd = EMD()
            emd.emd(df[feature][i:i+time_steps].values, max_imf=max_imf)
            try:
                imfs, residue = emd.get_imfs_and_residue()
                if (max(residue) == 0) & (min(residue) == 0):
                    imfs = np.roll(imfs, 1, axis=0)
                    residue = imfs[0].copy()
                imfs[0] = np.zeros(imfs.shape[1])
                columns = ['{}_IMF{}'.format(feature, i) for i in range(1, imfs.shape[0] + 1)]
                result += [pd.concat([cur_time, pd.DataFrame(imfs.T, columns=columns), pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]
            except:
                skip.append(i)

    save_file = Path('./EMD_{}/skip_list.pkl'.format(time_steps))
    with open(save_file, 'wb') as f:
        pickle.dump(skip, f)

    for feature in features:
        save_file = Path('./EMD_{}/{}_imf_list.pkl'.format(time_steps, feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            if i in skip:
                continue
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            emd = EMD()
            emd.emd(df[feature][i:i+time_steps].values, max_imf=max_imf)
            imfs, residue = emd.get_imfs_and_residue()
            if (max(residue) == 0) & (min(residue) == 0):
                imfs = np.roll(imfs, 1, axis=0)
                residue = imfs[0].copy()
            imfs[0] = np.zeros(imfs.shape[1])
            columns = ['{}_IMF{}'.format(feature, i) for i in range(1, imfs.shape[0] + 1)]
            result += [pd.concat([cur_time, pd.DataFrame(imfs.T, columns=columns), pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]

        with open(save_file, 'wb') as f:
            pickle.dump(result, f)


def create_dataset():

    load_data = dict()
    for feature in features:
        read_file = Path('./EMD_{}/{}_imf_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = df['close'][time_steps:train_size+time_steps].values
    for feature in features:
        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size):
            insuf = max_imf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./EMD_{}/train.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['close'][train_size + time_steps:].values
    for feature in features:
        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_imf - load_data[feature][i + train_size].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size][c].values)
                residue += 1

        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./EMD_{}/test.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_binary():

    load_data = dict()
    for feature in features:
        read_file = Path('./EMD_{}/{}_imf_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    labels = np.sign(np.diff(df['adj_close'].values)).astype(int)
    train_size = len(load_data['adj_close']) - test_size

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size].values
    train_data['y'] = labels[time_steps-1:train_size-1]
    for feature in features:
        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
            insuf = max_imf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./EMD_{}/train_binary.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size:].values
    test_data['y'] = labels[train_size-1:]
    for feature in features:
        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_imf - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
                residue += 1

        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./EMD_{}/test_binary.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_multiclass(hold_days=5):
    '''
        lebel:
            0 -> nop_up 不動作，但股票漲
            1 -> long 做多
            2 -> short 做空


        return rate:
            spread[i] = open[i] - close[i + hold_days]
    '''

    _open, _close = df['adj_open'].values[: -(hold_days-1)], np.roll(df['adj_close'].values, -(hold_days-1))[: -(hold_days-1)]
    fee = (_open * 1.425 * 1e-3 + _close * 4.425 * 1e-3)
    long = (_close - _open) - fee > 0
    short = (_open - _close) - fee > 0
    nop = ~(long | short)
    acts = [nop, long, short]

    load_data = dict()
    for feature in features:
        read_file = Path('./EMD_{}/{}_imf_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    with open(Path('./EMD_{}/skip_list.pkl'.format(time_steps)), 'rb') as f:
        skip = pickle.load(f)
    train_size = len(load_data['adj_close']) - len(skip) - test_size + time_steps
    acts = np.delete(acts, skip, 0)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = np.delete(dates[time_steps: train_size].values, skip, 0)
    train_data['y'] = np.zeros(train_size - time_steps, dtype=int)
    train_data['spread_long'] = ((_close - _open) - fee)[time_steps: train_size]
    train_data['spread_short'] = ((_open - _close) - fee)[time_steps: train_size]

    for i, act in enumerate(acts):
        train_data['y'][act[time_steps:train_size]] = i

    for feature in features:
        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
            insuf = max_imf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_imf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./EMD_{}/train_multiclass_hold_{}.pkl'.format(time_steps, hold_days))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f, protocol=4)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size:-(hold_days-1)].values
    test_data['y'] = np.zeros(test_size-(hold_days-1), dtype=int)
    test_data['spread_long'] = ((_close - _open) - fee)[train_size:]
    test_data['spread_short'] = ((_open - _close) - fee)[train_size:]

    for i, act in enumerate(acts):
        test_data['y'][act[train_size:]] = i

    for feature in features:
        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size - (hold_days - 1)):
            insuf = max_imf - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
                residue += 1

        for i in range(max_imf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./EMD_{}/test_multiclass_hold_{}.pkl'.format(time_steps, hold_days))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f, protocol=4)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./EMD/train.pkl', 'rb') as f:
        a = pickle.load(f)
    with open('./EMD/test.pkl', 'rb') as f:
        b = pickle.load(f)
    '''

    # compute and save emd result
    # to_emd()

    # create training set and data set by day
    # create_dataset()
    # create_dataset_binary()
    create_dataset_multiclass()
