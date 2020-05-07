import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from pykalman import KalmanFilter
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from lib.attention2 import Attention


class Load():
    def __init__(self, model_path, wavelet, plot=True):
        self.model_path = model_path
        self.wavelet = wavelet
        self.plot = plot
        self.load_data()

    def load_data(self):
        with open(Path('./{}/train.pkl'.format(self.wavelet)), 'rb') as f:
            self.train_X = pickle.load(f)
        with open(Path('./{}/test.pkl'.format(self.wavelet)), 'rb') as f:
            self.test_X = pickle.load(f)

        with open(Path('./{}/train_binary.pkl'.format(self.wavelet)), 'rb') as f:
            self.binary_train_y = pickle.load(f).pop('y', None)
        with open(Path('./{}/test_binary.pkl'.format(self.wavelet)), 'rb') as f:
            self.binary_test_y = pickle.load(f).pop('y', None)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)

    def load_model(self):
        self.model = load_model(self.model_path, custom_objects={'Attention': Attention})

    def test(self, skip_days=4000):
        '''
        input data create & normalization
        '''
        inputTrain_X = [v for v in self.train_X.values()]
        inputTest_X = [v for v in self.test_X.values()]

        scaler_X = MinMaxScaler(feature_range=(0, 1))
        for i in range(len(inputTrain_X)):
            inputTrain_X[i] = scaler_X.fit_transform(inputTrain_X[i])[:, :, np.newaxis]
            inputTest_X[i] = scaler_X.transform(inputTest_X[i])[:, :, np.newaxis]

        self.scaler_Y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_Y.fit(self.train_Y.reshape(-1, 1))

        '''
        price predict
        '''
        trainPredict = self.scaler_Y.inverse_transform(self.model.predict(inputTrain_X))[:, 0]
        testPredict = self.scaler_Y.inverse_transform(self.model.predict(inputTest_X))[:, 0]

        '''
        plot loss curve & price curve
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
        error estimate
        '''
        ein_rmse = np.sqrt(mean_squared_error(self.train_Y, trainPredict))
        eout_rmse = np.sqrt(mean_squared_error(self.test_Y, testPredict))
        ein_r2 = r2_score(self.train_Y, trainPredict)
        eout_r2 = r2_score(self.test_Y, testPredict)
        print('[{}] ----------> \tEin: {:.8f} \tEout: {:.8f} \tR^2_in: {:.8f} \tR^2_out: {:.8f}\n'.format(self.wavelet, ein_rmse, eout_rmse, ein_r2, eout_r2))

        trainPredict_binary = np.sign(np.diff(trainPredict)).astype(int)
        print('(train) trend accuracy =', sum(trainPredict_binary == self.binary_train_y[1:])/len(trainPredict_binary))
        testPredict_binary = np.sign(np.diff(testPredict)).astype(int)
        print('(test) trend accuracy =', sum(testPredict_binary == self.binary_test_y[1:])/len(testPredict_binary))


if __name__ == "__main__":
    model_name = 'attention_2_haar_64'
    model = Load(model_path=Path('model/checkpoint/{}.h5'.format(model_name)), wavelet='haar_64')
    model.load_model()
    model.test()
