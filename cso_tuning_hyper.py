from lib.cso import cso
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
from for_optimize_LSTM import opt_LSTM
from for_optimize_LSTM_binary import opt_binary_LSTM
from for_optimize_LSTM_multiclass import opt_multi_LSTM
import numpy as np
from pathlib import Path


wavelet = 'pure_16'
minimize = False


def fittness(x):
    model = opt_LSTM(denoise=True, wavelet=wavelet, plot=False)
    return model.train_test(lstm_units=int(x[0]), lstm_act_f=int(x[1]), lstm_layer=int(x[2]), lstm_dropout=x[3], lstm_recurrent_dropout=x[4], att=int(x[5]),
                            dense_unit=int(x[6]), dense_layer=int(x[7]), dense_act_f=int(x[8]), dense_drop=x[9], batch_size=int(x[10]), epochs=600, save_model=True)


def fittness_binary(x):
    try:
        model = opt_binary_LSTM(denoise=False, wavelet=wavelet, plot=False)
        return model.train_test(bidirect=int(x[0]), rec_layer=int(x[1]), lstm_l2=x[2], lstm_units=int(x[3]), lstm_layer=int(x[4]), lstm_dropout=x[5], lstm_recurrent_dropout=x[6],
                                dense_l2=x[7], dense_unit=int(x[8]), dense_layer=int(x[9]), dense_act_f=int(x[10]), dense_drop=x[11], BatchNorm=int(x[12]), batch_size=int(x[13]), epochs=1000, save_model=True)
    except KeyboardInterrupt:
        print("W: interrupt received, stopping…")
        raise
    except:
        return 0


def fittness_multiclass(x):
    try:
        model = opt_multi_LSTM(denoise=False, wavelet=wavelet, plot=False)
        return model.train_test(bidirect=int(x[0]), rec_layer=int(x[1]), lstm_l2=x[2], lstm_units=int(x[3]), lstm_layer=int(x[4]), lstm_dropout=x[5], lstm_recurrent_dropout=x[6],
                                dense_l2=x[7], dense_unit1=int(x[8]), dense_unit2=int(x[9]), dense_unit3=int(x[10]), dense_act_f=int(x[11]), dense_drop=x[12], BatchNorm=int(x[13]), batch_size=int(x[14]), epochs=1000, save_model=True)
    except KeyboardInterrupt:
        print("W: interrupt received, stopping…")
        raise
    except:
        return 0


'''
def __init__(self, denoise=None, wavelet='OriginData', plot=True):
def train_test(self, bidirect=True, rec_layer=0, lstm_l1=1e-5, lstm_l2=1e-5, lstm_units=100, lstm_layer=1, lstm_act_f=0, lstm_dropout=0.0, lstm_recurrent_dropout=0.0, att=0,
                dense_l1=1e-5, dense_l2=1e-5, dense_unit=32, dense_layer=1, dense_act_f=0, dense_drop=0.0, BatchNorm=True, batch_size=160, epochs=600, save_model=False):
[bidirect, rec_layer, lstm_l2, lstm_units, lstm_layer, lstm_dropout, lstm_recurrent_dropout, dense_l2, dense_unit, dense_layer, dense_act_f, dense_drop, BatchNorm, batch_size]
'''

# if minimize:
#     alh = cso(wavelet, 10, fittness, [64, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80],
#               [512, 2, 4, 0.3, 0.3, 4, 64, 10, 5,  0.3, 300],
#               11, 30, pa=0.25, nest=50, discrete=[True, True, True, False, False, True, True, True, True, False, True])
# else:
#       alh = cso(wavelet, 10, fittness_binary, [0, 0, 0, 64, 1, 0, 0, 0, 8, 0, 0, 0, 0, 80],
#               [1, 1,  1e-3/2, 256, 3, 0.25, 0.25, 1e-3/2, 128, 3, 4,  0.25, 1, 400],
#               14, 30, pa=0.25, nest=50, discrete=[True, True, False, True, True, False, False, False, True, True, True, False, True, True], minimize=minimize)

alh = cso(wavelet, 10, fittness_multiclass, [0, 0, 0, 64, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 80],
          [1, 1,  1e-3/2, 256, 3, 0.25, 0.25, 1e-3/2, 128, 128, 128, 4, 0.25, 1, 400],
          15, 30, pa=0.25, nest=50, discrete=[True, True, False, True, True, False, False, False, True, True, True, True, False, True, True], minimize=minimize)

if minimize:
    save_log = Path('./log/price/{}'.format(wavelet))
    save_log.mkdir(exist_ok=True)
    save_log = save_log / 'bestparam_log.txt'
    print('{} loss: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
    with open(save_log, 'a+') as f:
        f.write('{} loss: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
else:
    save_log = Path('./log/multiclass/{}'.format(wavelet))
    save_log.mkdir(exist_ok=True)
    save_log = save_log / 'bestparam_log.txt'
    print('{} acc: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
    with open(save_log, 'a+') as f:
        f.write('{} acc: {}\n {}\n\n'.format(wavelet, alh.get_best_fitness(), alh.get_Gbest()))
