import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pathlib import Path
import pickle


class wavelet_LSTM():
    def __init__(self, denoise=None, wavelet='pure_64', plot=True):
        self.wavelet = wavelet
        self.plot = plot
        self.denoise = denoise
        self.load_data()

    def load_data(self):

        if self.denoise:
            with open(Path('./{}/denoise/train.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/denoise/test.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)
        else:
            with open(Path('./{}/train.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/test.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)

    def train_test_loop(self, units=10, act_f=0, N_layer=1, lr=0.1, dropout=0.0, recurrent_dropout=0.0, epochs=600, batch_size=180, save_model=False):
        '''
        model create
        '''
        model_input = []
        model_output = []
        for _key in self.train_X.keys():
            m = Sequential()
            for _layer in range(N_layer-1):
                m.add(LSTM(units=units,  activation='tanh', dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True, input_shape=(None, 1)))
            m.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(None, 1)))
            model_input += [m.input]
            model_output += [m.output]

        x = Concatenate()(model_output)
        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)
        model = Model(model_input, x)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

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
        inputTrain_Y = self.scaler_Y.fit_transform(self.train_Y.reshape(-1, 1))[:, 0]
        inputTest_Y = self.scaler_Y.transform(self.test_Y.reshape(-1, 1))[:, 0]

        '''
        model train
        '''
        if self.denoise:
            get_best_model = ModelCheckpoint('model/checkpoint/pure_lstm_denoise_{}.h5'.format(self.wavelet), monitor='val_loss', save_best_only=True)
        else:
            get_best_model = ModelCheckpoint('model/checkpoint/pure_lstm_{}.h5'.format(self.wavelet), monitor='val_loss', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, mode='auto', verbose=1)
        hLS = model.fit(inputTrain_X, inputTrain_Y, validation_data=(inputTest_X, inputTest_Y), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr, get_best_model], verbose=2)

        if save_model:
            model.save('model/pure_lstm_{}.h5'.format(self.wavelet))
        '''
        price predict
        '''
        trainPredict = self.scaler_Y.inverse_transform(model.predict(inputTrain_X))[:, 0]
        testPredict = self.scaler_Y.inverse_transform(model.predict(inputTest_X))[:, 0]

        '''
        plot loss curve & price curve
        '''
        plt.figure()
        plt.plot(range(epochs), hLS.history['loss'], color='blue', label='loss')
        plt.plot(range(epochs), hLS.history['val_loss'], color='red', label='val_loss')
        plt.xlabel('epoch')
        plt.legend()
        if self.denoise:
            plt.savefig(Path('losscurve/pure_lstm_denoise_{}.png'.format(self.wavelet)))
        else:
            plt.savefig(Path('losscurve/pure_lstm_{}.png'.format(self.wavelet)))

        if self.plot:
            real_date = np.concatenate((self.train_date, self.test_date), axis=0)
            real_price = np.concatenate((self.train_Y, self.test_Y), axis=0)
            plt.subplot(211)
            plt.plot_date(real_date, real_price, '-', color='black', label='real')
            plt.plot_date(self.train_date, trainPredict, '-', color='blue', label='predict(train)')
            plt.plot_date(self.test_date, testPredict, '-', color='red', label='predict(test)')
            plt.xlabel('days')
            plt.ylabel('price')
            plt.legend()

            plt.subplot(212)
            plt.plot_date(self.train_date, abs(self.train_Y - trainPredict), '-', color='blue', label='predict(train)')
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
        print('[pure_LSTM] ----------> \tEin: {:.8f} \tEout: {:.8f} \tR^2_in: {:.8f} \tR^2_out: {:.8f}\n'.format(ein_rmse, eout_rmse, ein_r2, eout_r2))


if __name__ == "__main__":
    for w in ['coif3_64', 'db3_64', 'haar_64', 'sym3_64']:
        model = wavelet_LSTM(denoise=False, wavelet=w, plot=False)
        model.train_test_loop()
