import pandas as pd
import numpy as np
import pywt
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
max_level = 3
wavelet = ['coif3', 'db3', 'haar', 'sym3'][3]


def to_wavelet():
    '''
    How to read pickle file in python?
    with open('./{}_{}/open_list.pkl'.format(wavelet, time_steps), 'rb') as f:
        data = pickle.load(f)
    '''
    for feature in features:
        save_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        result = list()
        for i in range(df.shape[0] - time_steps):
            cur_time = dates[i:i+time_steps].copy().reset_index(drop=True)
            coeffs = pywt.swt(df[feature][i:i+time_steps].values, wavelet=wavelet, trim_approx=True, norm=True, level=max_level)
            columns = ['{}_cD{}'.format(feature, i) for i in range(1, len(coeffs))]
            result += [pd.concat([cur_time,  pd.DataFrame(coeffs[0], columns=['{}_cA'.format(feature)]), pd.DataFrame(np.array(coeffs[1:]).T, columns=columns)], axis=1)]

        print('{} max: {} min: {}'.format(feature, max([i.shape[1] for i in result]), min([i.shape[1] for i in result])))
        with open(save_file, 'wb') as f:
            pickle.dump(result, f)


def create_dataset():

    load_data = dict()
    for feature in features:
        read_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = df['close'][time_steps:train_size + time_steps].values
    for feature in features:
        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size):
            insuf = max_level - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./{}_{}/train.pkl'.format(wavelet, time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['close'][train_size + time_steps:].values
    for feature in features:
        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_level - load_data[feature][i + train_size].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size][c].values)
                residue += 1

        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./{}_{}/test.pkl'.format(wavelet, time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_denoise():

    load_data = dict()
    for feature in features:
        read_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    denoise_dir = Path('./{}_{}/denoise/'.format(wavelet, time_steps))
    denoise_dir.mkdir(exist_ok=True)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = df['close'][time_steps:train_size+time_steps].values
    for feature in features:
        c = load_data[feature][0].columns[1]
        train_data[feature] = np.array([load_data[feature][i][c].values for i in range(train_size)])

    train_file = Path('train.pkl')
    with open(denoise_dir / train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['close'][train_size + time_steps:].values
    for feature in features:
        c = load_data[feature][0].columns[1]
        test_data[feature] = np.array([load_data[feature][i + train_size][c].values for i in range(test_size)])

    test_file = Path('test.pkl')
    with open(denoise_dir / test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_binary():

    load_data = dict()
    for feature in features:
        read_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = np.sign(np.diff(df['close'][time_steps-1:train_size + time_steps].values)).astype(int)
    for feature in features:
        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size):
            insuf = max_level - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./{}_{}/train_binary.pkl'.format(wavelet, time_steps))
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = np.sign(np.diff(df['close'][train_size-1 + time_steps:].values)).astype(int)
    for feature in features:
        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_level - load_data[feature][i + train_size].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size][c].values)
                residue += 1

        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./{}_{}/test_binary.pkl'.format(wavelet, time_steps))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_denoise_binary():

    load_data = dict()
    for feature in features:
        read_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    denoise_dir = Path('./{}_{}/denoise/'.format(wavelet, time_steps))
    denoise_dir.mkdir(exist_ok=True)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size + time_steps].values
    train_data['y'] = np.sign(np.diff(df['close'][time_steps-1:train_size + time_steps].values)).astype(int)
    for feature in features:
        c = load_data[feature][0].columns[1]
        train_data[feature] = np.array([load_data[feature][i][c].values for i in range(train_size)])

    train_file = Path('train_binary.pkl')
    with open(denoise_dir / train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = np.sign(np.diff(df['close'][train_size-1 + time_steps:].values)).astype(int)
    for feature in features:
        c = load_data[feature][0].columns[1]
        test_data[feature] = np.array([load_data[feature][i + train_size][c].values for i in range(test_size)])

    test_file = Path('test_binary.pkl')
    with open(denoise_dir / test_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./{}_{}/train.pkl'.format(wavelet, time_steps), 'rb') as f:
        a = pickle.load(f)
    with open('./{}_{}/test.pkl'.format(wavelet, time_steps), 'rb') as f:
        b = pickle.load(f)
    '''

    # # compute and save emd result
    # to_wavelet()

    # # create training set and data set by day
    # create_dataset()
    # create_dataset_denoise()
    create_dataset_binary()
    create_dataset_denoise_binary()
