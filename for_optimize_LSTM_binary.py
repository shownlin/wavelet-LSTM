import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import GRU, LSTM, Dense, Concatenate, TimeDistributed, Flatten, Input, Dropout, BatchNormalization, Activation, Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from lib.attention_withcontext import AttentionWithContext
from lib.attention2 import Attention
from lib.attention3 import AttentionWeightedAverage
from lib.attention import attention_3d_block
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pathlib import Path
import pickle


class opt_binary_LSTM():
    def __init__(self, denoise=None, wavelet='pure_16', plot=True):
        self.wavelet = wavelet
        self.plot = plot
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

    def train_test(self, bidirect=True, rec_layer=0, lstm_l1=1e-5, lstm_l2=1e-5, lstm_units=100, lstm_layer=1, lstm_act_f=0, lstm_dropout=0.0, lstm_recurrent_dropout=0.0, att=0,
                   dense_l1=1e-5, dense_l2=1e-5, dense_unit=32, dense_layer=1, dense_act_f=0, dense_drop=0.0, BatchNorm=True, batch_size=160, epochs=1000, save_model=False):
        '''
        model create
        '''
        rec_layer = GRU if rec_layer else LSTM
        model_input = []
        model_output = []
        lstm_act_f = ['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'][lstm_act_f]
        n_lstm = len(self.train_X.keys())
        lstm_reg = l1_l2(l1=lstm_l1, l2=lstm_l2)
        if (not self.denoise) and ('pure' not in self.wavelet):
            n_lstm //= 6
        for _key in range(n_lstm):
            x = i = Input(shape=(self.time_step, 1)) if self.denoise else Input(shape=(self.time_step, 6))
            for _layer in range(lstm_layer - 1):
                if bidirect:
                    x = Bidirectional(rec_layer(units=lstm_units, activation=lstm_act_f, kernel_regularizer=lstm_reg,
                                                dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True))(x)
                else:
                    x = rec_layer(units=lstm_units, activation=lstm_act_f, kernel_regularizer=lstm_reg, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True)(x)
            # Apply a single dense layer to all timesteps of the resulting sequence to convert back to prices
            if att == 0:
                if bidirect:
                    x = Bidirectional(rec_layer(units=lstm_units, kernel_regularizer=lstm_reg, activation=lstm_act_f, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(x)
                else:
                    x = rec_layer(units=lstm_units, activation=lstm_act_f, kernel_regularizer=lstm_reg, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(x)
            else:
                if bidirect:
                    x = Bidirectional(rec_layer(units=lstm_units, activation=lstm_act_f, kernel_regularizer=lstm_reg,
                                                dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True))(x)
                else:
                    x = rec_layer(units=lstm_units, activation=lstm_act_f, kernel_regularizer=lstm_reg,  dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True)(x)
                if att == 1:
                    x = AttentionWithContext()(x)
                elif att == 2:
                    x = Attention(self.time_step)(x)
                elif att == 3:
                    x = AttentionWeightedAverage()(x)
                elif att == 4:
                    x = attention_3d_block(x)
            m = Model(i, x)
            model_input += [m.input]
            model_output += [m.output]

        x = Concatenate()(model_output)
        dense_act_f = ['relu', 'selu', 'tanh', 'elu', 'sigmoid', 'linear'][dense_act_f]
        dense_reg = l1_l2(l1=dense_l1, l2=dense_l2)
        for _layer in range(dense_layer):
            x = Dense(dense_unit, kernel_regularizer=dense_reg)(x)
            if BatchNorm:
                x = BatchNormalization()(x)
            x = Activation(activation=dense_act_f)(x)
            if dense_drop > 0:
                x = Dropout(dense_drop)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(model_input, x)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        print('rnn activation: {}, l1={:.8f} l2={:.8f}\tdense activation: {}, l1={:.8f} l2={:.8f}'.format(lstm_act_f, lstm_l1, lstm_l2, dense_act_f, dense_l1, dense_l2))

        '''
        input data create & normalization
        '''
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
        model train
        '''
        if self.denoise:
            checkpoint_path = './model/checkpoint/for_opt_denoise_binary_{}.h5'.format(self.wavelet)

        else:
            checkpoint_path = './model/checkpoint/for_opt_binary_{}.h5'.format(self.wavelet)
        get_best_model = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='auto', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='accuracy', patience=30, factor=0.8, mode='auto', verbose=1)
        hLS = model.fit(inputTrain_X, inputTrain_Y, validation_data=(inputTest_X, inputTest_Y), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr, get_best_model], verbose=2)

        '''
        trend predict
        '''
        if att == 1:
            custom_objects = {'AttentionWithContext': AttentionWithContext}
        elif att == 2:
            custom_objects = {'Attention': Attention}
        elif att == 3:
            custom_objects = {'AttentionWeightedAverage': AttentionWeightedAverage}
        else:
            custom_objects = None

        model = load_model(checkpoint_path, custom_objects=custom_objects)
        trainPredict = [0 if y <= 0.5 else 1 for y in model.predict(inputTrain_X).flatten()]
        testPredict = [0 if y <= 0.5 else 1 for y in model.predict(inputTest_X).flatten()]

        '''
        plot loss curve & price curve
        '''
        plt.figure()
        plt.plot(range(epochs), hLS.history['accuracy'], color='blue', label='accuracy')
        plt.plot(range(epochs), hLS.history['val_accuracy'], color='red', label='val_accuracy')
        plt.xlabel('epoch')
        plt.legend()
        if self.denoise:
            plt.savefig(Path('losscurve/for_opt_denoise_binary_{}.png'.format(self.wavelet)))
        else:
            plt.savefig(Path('losscurve/for_opt_binary_{}.png'.format(self.wavelet)))

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
        accuracy estimate
        '''
        acc_in = sum(trainPredict == inputTrain_Y) / len(trainPredict)
        acc_out = sum(testPredict == inputTest_Y) / len(testPredict)
        print('[cuckoo] ----------> \tAcc_in: {:.8f} \tAcc_out: {:.8f}\n'.format(acc_in, acc_out))

        '''
        save model
        '''
        save_dir = Path('./model/for_opt/{}'.format(self.wavelet))
        save_dir.mkdir(exist_ok=True)
        save_file = save_dir / '{}.h5'.format(acc_out)
        if save_model:
            model.save(save_file)

        return acc_out


if __name__ == "__main__":
    for w in ['haar_16']:
        lstm = opt_binary_LSTM(wavelet=w, plot=True)
        lstm.train_test()
