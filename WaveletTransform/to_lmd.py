import pandas as pd
import numpy as np
from lib.LMD import LMD
from pathlib import Path
import pickle

origin_dir = Path('./OriginData')
orgin_file = origin_dir / 'TX_price.csv'
df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
dates = df['date']
features = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
time_steps = 64
train_size = int(df.shape[0] * 0.94)
test_size = df.shape[0] - train_size - time_steps
max_num_pf = 3


def to_lmd():
    '''
    How to read pickle file in python?
    with open('./LMD/open_PF_list.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
    for feature in features:
        save_file = Path('./LMD_{}/{}_PF_list.pkl'.format(time_steps, feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            lmd = LMD(max_num_pf=max_num_pf)
            pfs, residue = lmd.lmd(df[feature][i:i+time_steps].values)
            columns = ['{}_PF{}'.format(feature, i) for i in range(1, len(pfs) + 1)]
            result += [pd.concat([cur_time, pd.DataFrame(pfs.T, columns=columns),  pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]

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
            insuf = max_num_pf - load_data[feature][i + train_size].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size][c].values)
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

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = np.sign(np.diff(df['close'][time_steps-1:train_size + time_steps].values)).astype(int)
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
    test_data['y'] = np.sign(np.diff(df['close'][train_size-1 + time_steps:].values)).astype(int)
    for feature in features:
        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_num_pf - load_data[feature][i + train_size].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size][c].values)
                residue += 1

        for i in range(max_num_pf+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./LMD_{}/test_binary.pkl'.format(time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./LMD/train.pkl', 'rb') as f:
        a = pickle.load(f)
    with open('./LMD/test.pkl', 'rb') as f:
        b = pickle.load(f)
    '''

    # compute and save emd result
    # to_lmd()

    # create training set and data set by day
    # create_dataset()
    create_dataset_binary()
