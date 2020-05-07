from lib.cso import cso
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
from for_optimize_LSTM import opt_LSTM
from for_optimize_LSTM_binary import opt_binary_LSTM
import numpy as np
from pathlib import Path


wavelet = 'pure_64'
minimize = False


def fittness(x):
    model = opt_LSTM(denoise=int(x[0]), wavelet=wavelet, plot=False)
    return model.train_test(lstm_units=int(x[1]), lstm_act_f=int(x[2]), lstm_layer=int(x[3]), lstm_dropout=x[4], lstm_recurrent_dropout=x[5], att=int(x[6]),
                            dense_unit=int(x[7]), dense_layer=int(x[8]), dense_act_f=int(x[9]), dense_drop=x[10], batch_size=int(x[11]), epochs=600, save_model=True)


def fittness_binary(x):
    model = opt_binary_LSTM(denoise=int(x[0]), wavelet=wavelet, plot=False)
    return model.train_test(lstm_units=int(x[1]), lstm_act_f=int(x[2]), lstm_layer=int(x[3]), lstm_dropout=x[4], lstm_recurrent_dropout=x[5], att=int(x[6]),
                            dense_unit=int(x[7]), dense_layer=int(x[8]), dense_act_f=int(x[9]), dense_drop=x[10], batch_size=int(x[11]), epochs=1000, save_model=True)


'''
def __init__(self, denoise=None, wavelet='OriginData', plot=True):
def train_test(self, lstm_units=10, lstm_act_f=0, lstm_layer=1, lstm_dropout=0.0, lstm_recurrent_dropout=0.0, att=0,
                   dense_unit=32, dense_layer=1, dense_act_f=0, dense_drop=0.0, batch_size=160, epochs=600, save_model=False):
[denoise, lstm_units, lstm_act_f, lstm_layer, lstm_dropout, lstm_recurrent_dropout, att, dense_unit, dense_layer, dense_act_f, dense_drop, batch_size]
'''

if minimize:
    alh = cso(wavelet, 10, fittness, [0, 64, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80],
              [0, 512, 3, 4, 0.3, 0.3, 4, 64, 10, 5,  0.3, 300],
              12, 30, pa=0.25, nest=50, discrete=[True, True, True, True, False, False, True, True, True, True, False, True])
else:
    alh = cso(wavelet, 10, fittness_binary, [0, 64, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80],
              [0, 512, 3, 4, 0.3, 0.3, 4, 64, 10, 5,  0.3, 300],
              12, 30, pa=0.25, nest=50, discrete=[True, True, True, True, False, False, True, True, True, True, False, True], minimize=minimize)

if minimize:
    save_log = Path('./log/price/{}'.format(wavelet))
    save_log.mkdir(exist_ok=True)
    save_log = save_log / 'bestparam_log.txt'
    print('{} loss: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
    with open(save_log, 'a+') as f:
        f.write('{} loss: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
else:
    save_log = Path('./log/binary/{}'.format(wavelet))
    save_log.mkdir(exist_ok=True)
    save_log = save_log / 'bestparam_log.txt'
    print('{} acc: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
    with open(save_log, 'a+') as f:
        f.write('{} acc: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
