import pandas as pd
from PyEMD import EMD
import numpy as np
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
max_imf = 2


def to_emd():
    '''
    How to read pickle file in python?
    with open('./EMD/open_imf_list.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
    for feature in features:
        save_file = Path('./EMD/{}_imf_list.pkl'.format(feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            emd = EMD()
            emd.emd(df[feature][i:i+time_steps].values, max_imf=max_imf)
            imfs, residue = emd.get_imfs_and_residue()
            if (max(residue) == 0) & (min(residue) == 0):
                imfs = np.roll(imfs, 1, axis=0)
                residue = imfs[0].copy()
                imfs[0] = np.zeros(imfs.shape[1])
            columns = ['{}_IMF{}'.format(feature, i) for i in range(1, imfs.shape[0] + 1)]
            result += [pd.concat([cur_time, pd.DataFrame(imfs.T, columns=columns),  pd.DataFrame(residue, columns=['{}_res'.format(feature)])], axis=1)]

        with open(save_file, 'wb') as f:
            pickle.dump(result, f)


def create_dataset():

    load_data = dict()
    for feature in features:
        read_file = Path('./EMD/{}_imf_list.pkl'.format(feature))
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

    train_file = Path('./EMD/train.pkl')
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

    test_file = Path('./EMD/test.pkl')
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./EMD/train.pkl', 'rb') as f:
        a = pickle.load(f)
    with open('./EMD/test.pkl', 'rb') as f:
        b = pickle.load(f)
    '''

    # compute and save emd result
    to_emd()

    # create training set and data set by day
    create_dataset()
