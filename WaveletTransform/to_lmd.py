import pandas as pd
import numpy as np
from lib.LMD import LMD
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
max_num_pf = 3


def to_lmd():
    '''
    How to read pickle file in python?
    with open('./LMD/open_PF_list.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
    skip = list()
    for feature in features:
        result = list()
        for i in range(df.shape[0] - time_steps):
            if i in skip:
                continue
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            lmd = LMD(max_num_pf=max_num_pf)
            try:
                pfs, residue = lmd.lmd(df[feature][i:i+time_steps].values)
                columns = ['{}_PF{}'.format(feature, i) for i in range(1, len(pfs) + 1)]
                result += [pd.concat([cur_time, pd.DataFrame(pfs.T, columns=columns), pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]
            except:
                skip.append(i)

    save_file = Path('./LMD_{}/skip_list.pkl'.format(time_steps))
    with open(save_file, 'wb') as f:
        pickle.dump(skip, f)

    for feature in features:
        save_file = Path('./LMD_{}/{}_PF_list.pkl'.format(time_steps, feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            if i in skip:
                continue
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            lmd = LMD(max_num_pf=max_num_pf)
            pfs, residue = lmd.lmd(df[feature][i:i+time_steps].values)
            columns = ['{}_PF{}'.format(feature, i) for i in range(1, len(pfs) + 1)]
            result += [pd.concat([cur_time, pd.DataFrame(pfs.T, columns=columns), pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]
        print('{} max: {} min: {}'.format(feature, max([i.shape[1] for i in result]), min([i.shape[1] for i in result])))
        with open(save_file, 'wb') as f:
            pickle.dump(result, f)


def create_dataset():

    load_data = dict()
    for feature in features:
        read_file = Path('./LMD_{}/{}_PF_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = df['close'][time_steps:train_size+time_steps].values
    for feature in features:
        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
            insuf = max_num_pf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./LMD_{}/train.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['close'][train_size + time_steps:].values
    for feature in features:
        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_num_pf - load_data[feature][i + train_size-time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size-time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size-time_steps][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./LMD_{}/test.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_binary():

    load_data = dict()
    for feature in features:
        read_file = Path('./LMD_{}/{}_PF_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    labels = np.sign(np.diff(df['adj_close'].values)).astype(int)
    train_size = len(load_data['adj_close']) - test_size

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = labels[time_steps-1:train_size-1]
    for feature in features:
        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size):
            insuf = max_num_pf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./LMD_{}/train_binary.pkl'.format(time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = labels[train_size-1:]
    for feature in features:
        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_num_pf - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./LMD_{}/test_binary.pkl'.format(time_steps))
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

    load_data = dict()
    for feature in features:
        read_file = Path('./LMD_{}/{}_pf_list.pkl'.format(time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    with open(Path('./LMD_{}/skip_list.pkl'.format(time_steps)), 'rb') as f:
        skip = pickle.load(f)
    train_size = len(load_data['adj_close']) - test_size + time_steps
    # acts = np.delete(acts, skip, 0)

    _dates = np.delete(dates[time_steps:].values, skip, 0)

    _open, _close = df['adj_open'].values[: -(hold_days-1)], np.roll(df['adj_close'].values, -(hold_days-1))[: -(hold_days-1)]
    fee = (_open * 1.425 * 1e-3 + _close * 4.425 * 1e-3)
    long = np.delete((_close - _open) - fee > 0, skip, 0)
    short = np.delete((_open - _close) - fee > 0, skip, 0)
    nop = np.delete(~(long | short), skip, 0)
    acts = [nop, long, short]

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = _dates[:train_size]
    train_data['y'] = np.zeros(train_size, dtype=int)
    train_data['spread_long'] = ((_close - _open) - fee)[time_steps:time_steps+train_size]
    train_data['spread_short'] = ((_open - _close) - fee)[time_steps:time_steps+train_size]

    for i, act in enumerate(acts):
        train_data['y'][act[:train_size]] = i

    for feature in features:
        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
            insuf = max_num_pf - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./LMD_{}/train_multiclass_hold_{}.pkl'.format(time_steps, hold_days))
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
        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size - (hold_days - 1)):
            insuf = max_num_pf - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./LMD_{}/test_multiclass_hold_{}.pkl'.format(time_steps, hold_days))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f, protocol=4)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./LMD/train.pkl', 'rb') as f:
        a = pickle.load(f)
    with open('./LMD/test.pkl', 'rb') as f:
        b = pickle.load(f)
    '''

    # compute and save LMD result
    # to_lmd()

    # create training set and data set by day
    # create_dataset()
    # create_dataset_binary()
    create_dataset_multiclass()
