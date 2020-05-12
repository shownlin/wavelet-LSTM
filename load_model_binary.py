import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from pykalman import KalmanFilter
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from lib.attention_withcontext import AttentionWithContext
from lib.attention2 import Attention
from lib.attention3 import AttentionWeightedAverage
from lib.attention import attention_3d_block


class Load():
    def __init__(self, model_path, wavelet, att, denoise=False, plot=True):
        self.model_path = model_path
        self.wavelet = wavelet
        self.plot = plot
        self.att = att
        self.denoise = denoise
        self.load_data()

    def load_data(self):
        if self.denoise:
            with open(Path('./{}/denoise/train_binary.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/denoise/test_binary.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)
        else:
            with open(Path('./{}/train_binary.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/test_binary.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)

    def load_model(self):
        if self.att == 1:
            custom_objects = {'AttentionWithContext': AttentionWithContext}
        elif self.att == 2:
            custom_objects = {'Attention': Attention}
        elif self.att == 3:
            custom_objects = {'AttentionWeightedAverage': AttentionWeightedAverage}
        else:
            custom_objects = None
        self.model = load_model(self.model_path, custom_objects=custom_objects)

    def test(self, skip_days=4000):
        '''
        input data create & normalization
        '''
        n_lstm = len(self.train_X.keys())
        if (not self.denoise) and ('pure' not in self.wavelet):
            n_lstm //= 6

        scaler_X = MinMaxScaler(feature_range=(0, 1))
        if (not self.denoise) and ('pure' not in self.wavelet):
            inputTrain_X = [[v[1] for v in self.train_X.items() if str(i) in v[0]] for i in range(1, n_lstm + 1)]
            inputTest_X = [[v[1] for v in self.test_X.items() if str(i) in v[0]] for i in range(1, n_lstm + 1)]
            for i in range(len(inputTrain_X)):
                for j in range(6):
                    inputTrain_X[i][j] = scaler_X.fit_transform(inputTrain_X[i][j])
                    inputTest_X[i][j] = scaler_X.transform(inputTest_X[i][j])
            inputTrain_X = np.moveaxis(inputTrain_X, 1, -1).tolist()
            inputTest_X = np.moveaxis(inputTest_X, 1, -1).tolist()
        else:
            inputTrain_X = [v for v in self.train_X.values()]
            inputTest_X = [v for v in self.test_X.values()]
            for i in range(len(inputTrain_X)):
                inputTrain_X[i] = scaler_X.fit_transform(inputTrain_X[i])[:, :, np.newaxis]
                inputTest_X[i] = scaler_X.transform(inputTest_X[i])[:, :, np.newaxis]

        inputTrain_Y = np.array([0 if y < 0 else y for y in self.train_Y])
        inputTest_Y = np.array([0 if y < 0 else y for y in self.test_Y])

        '''
        price predict
        '''
        trainPredict = [0 if y <= 0.5 else 1 for y in self.model.predict(inputTrain_X).flatten()]
        testPredict = [0 if y <= 0.5 else 1 for y in self.model.predict(inputTest_X).flatten()]

        '''
        plot loss curve & price curve
        '''
        '''
        if self.plot:
            real_date = np.concatenate((self.train_date, self.test_date), axis=0)[skip_days:]
            real_price = np.concatenate((self.train_Y, self.test_Y), axis=0)[skip_days:]
            plt.subplot(211)
            plt.plot_date(real_date, real_price, '-', color='black', label='real')
            plt.plot_date(self.train_date[skip_days:], trainPredict[skip_days:], '-', color='blue', label='predict(train)')
            plt.plot_date(self.test_date, testPredict, '-', color='red', label='predict(test)')
            plt.xlabel('days')
            plt.ylabel('price')
            plt.legend()

            plt.subplot(212)
            plt.plot_date(self.train_date[skip_days:], abs(self.train_Y - trainPredict)[skip_days:], '-', color='blue', label='predict(train)')
            plt.plot_date(self.test_date, abs(self.test_Y - testPredict), '-', color='red', label='predict(test)')
            plt.xlabel('days')
            plt.ylabel('bias')
            plt.legend()
            plt.show()
        '''

        '''
        accuracy estimate
        '''
        acc_in = sum(trainPredict == inputTrain_Y) / len(trainPredict)
        acc_out = sum(testPredict == inputTest_Y) / len(testPredict)
        print('Train accuracy = {}, Test accuracy = {}'.format(acc_in, acc_out))


if __name__ == "__main__":
    model_name = 'for_opt_binary_haar_16'
    model = Load(model_path=Path('model/checkpoint/{}.h5'.format(model_name)), wavelet='haar_16', att=2)
    model.load_model()
    model.test()
