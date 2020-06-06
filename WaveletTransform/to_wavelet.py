import pandas as pd
import numpy as np
import pywt
from pathlib import Path
import pickle

origin_dir = Path('./OriginData')
orgin_file = origin_dir / 'TX_price.csv'
df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
dates = df['date']
features = ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
test_size = 250
train_size = df.shape[0] - test_size
time_steps = 16
max_level = pywt.swt_max_level(time_steps)
# wavelet = ['coif3', 'db3', 'haar', 'sym3'][0]


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
    train_data['y'] = df['adj_close'][time_steps:train_size + time_steps].values
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
    test_data['y'] = df['adj_close'][train_size + time_steps:].values
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
    train_data['y'] = df['adj_close'][time_steps:train_size+time_steps].values
    for feature in features:
        c = load_data[feature][0].columns[1]
        train_data[feature] = np.array([load_data[feature][i][c].values for i in range(train_size)])

    train_file = Path('train.pkl')
    with open(denoise_dir / train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size + time_steps:].values
    test_data['y'] = df['adj_close'][train_size + time_steps:].values
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

    labels = np.sign(np.diff(df['adj_close'].values)).astype(int)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size].values
    train_data['y'] = labels[time_steps-1:train_size-1]
    for feature in features:
        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
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
    test_data['date'] = dates[train_size:].values
    test_data['y'] = labels[train_size-1:]
    for feature in features:
        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size):
            insuf = max_level - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
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

    labels = np.sign(np.diff(df['adj_close'].values)).astype(int)

    denoise_dir = Path('./{}_{}/denoise/'.format(wavelet, time_steps))
    denoise_dir.mkdir(exist_ok=True)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size].values
    train_data['y'] = labels[time_steps-1:train_size-1]
    for feature in features:
        c = load_data[feature][0].columns[1]
        train_data[feature] = np.array([load_data[feature][i][c].values for i in range(train_size - time_steps)])

    train_file = Path('train_binary.pkl')
    with open(denoise_dir / train_file, 'wb') as f:
        pickle.dump(train_data, f)

    # testing set
    test_data = dict()
    test_data['date'] = dates[train_size:].values
    test_data['y'] = labels[train_size-1:]
    for feature in features:
        c = load_data[feature][0].columns[1]
        test_data[feature] = np.array([load_data[feature][i + train_size - time_steps][c].values for i in range(test_size)])

    test_file = Path('test_binary.pkl')
    with open(denoise_dir / test_file, 'wb') as f:
        pickle.dump(test_data, f)


def create_dataset_multiclass(hold_days=5):
    '''
        lebel:
            0 -> long 做多
            1 -> short 做空
            2 -> nop_up 不動作，但股票漲
            3 -> nop_down 不動作，但股票跌

        return rate:
            spread[i] = open[i] - close[i + hold_days]
    '''

    _open, _close = df['open'].values[: - (hold_days - 1)], np.roll(df['close'].values, -(hold_days - 1))[: - (hold_days - 1)]
    fee = (_open * 1.425 * 1e-3 + _close * 4.425 * 1e-3)
    long = (_close - _open) - fee > 0
    short = (_open - _close) - fee > 0
    # nop_up = ~(long | short) & (_close > _open)
    # nop_down = ~(long | short) & (_open > _close)
    # acts = [long, short, nop_up, nop_down]
    nop = ~(long | short)
    acts = [long, short, nop]

    load_data = dict()
    for feature in features:
        read_file = Path('./{}_{}/{}_list.pkl'.format(wavelet, time_steps, feature))
        with open(read_file, 'rb') as f:
            load_data[feature] = pickle.load(f)

    # training set
    train_data = dict()
    train_data['time_step'] = time_steps
    train_data['date'] = dates[time_steps:train_size].values
    train_data['y'] = np.zeros(train_size - time_steps, dtype=int)
    train_data['spread_long'] = ((_close - _open) - fee)[time_steps:train_size]
    train_data['spread_short'] = ((_open - _close) - fee)[time_steps:train_size]

    for i, act in enumerate(acts):
        train_data['y'][act[time_steps:train_size]] = i

    for feature in features:
        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(train_size - time_steps):
            insuf = max_level - load_data[feature][i].shape[1] + 2
            for j in range(insuf):
                train_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i].columns[1:]:
                train_data['{}_{}'.format(feature, residue)].append(load_data[feature][i][c].values)
                residue += 1

        for i in range(max_level+1):
            train_data['{}_{}'.format(feature, i + 1)] = np.array(train_data['{}_{}'.format(feature, i + 1)])

    train_file = Path('./{}_{}/train_multiclass_hold_{}.pkl'.format(wavelet, time_steps, hold_days))
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
        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i+1)] = list()

        for i in range(test_size - (hold_days - 1)):
            insuf = max_level - load_data[feature][i + train_size - time_steps].shape[1] + 2
            for j in range(insuf):
                test_data['{}_{}'.format(feature, j+1)].append(np.zeros(time_steps))

            residue = insuf+1
            for c in load_data[feature][i + train_size - time_steps].columns[1:]:
                test_data['{}_{}'.format(feature, residue)].append(load_data[feature][i + train_size - time_steps][c].values)
                residue += 1

        for i in range(max_level+1):
            test_data['{}_{}'.format(feature, i + 1)] = np.array(test_data['{}_{}'.format(feature, i + 1)])

    test_file = Path('./{}_{}/test_multiclass_hold_{}.pkl'.format(wavelet, time_steps, hold_days))
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f, protocol=4)


if __name__ == "__main__":
    '''
    How to read the traing set and the testing set?
    with open('./{}_{}/train.pkl'.format(wavelet, time_steps), 'rb') as f:
        a = pickle.load(f)
    with open('./{}_{}/test.pkl'.format(wavelet, time_steps), 'rb') as f:
        b = pickle.load(f)
    '''

    for wavelet in ['coif3', 'db3', 'haar', 'sym3']:

        # compute and save emd result
        to_wavelet()

        # create training set and data set by day
        # create_dataset()
        # create_dataset_denoise()
        create_dataset_binary()
        # create_dataset_denoise_binary()
        # create_dataset_multiclass()
