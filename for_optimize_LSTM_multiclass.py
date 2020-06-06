import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import GRU, LSTM, Dense, Concatenate, TimeDistributed, Flatten, Input, Dropout, BatchNormalization, Activation, Bidirectional, LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from lib.custom_callback import CustomCheckpoint_multiclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pathlib import Path
import pickle
from keras import backend as K


def long_short_metric(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)
    mask1 = K.equal(y_true, K.constant(0, dtype=y_true.dtype))
    mask2 = K.equal(y_true, K.constant(1, dtype=y_true.dtype))
    mask = tf.math.logical_or(mask1, mask2)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return K.cast(K.equal(y_true, y_pred), K.floatx())


class opt_multi_LSTM():
    def __init__(self, denoise=None, wavelet='pure_20', hold=5, plot=True):
        self.wavelet = wavelet
        self.plot = plot
        self.hold = hold
        self.denoise = denoise
        self.load_data()

    def load_data(self):
        if self.denoise:
            with open(Path('./{}/denoise/train_multiclass_hold_{}.pkl'.format(self.wavelet, self.hold)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/denoise/test_multiclass_hold_{}.pkl'.format(self.wavelet, self.hold)), 'rb') as f:
                self.test_X = pickle.load(f)
        else:
            with open(Path('./{}/train_multiclass_hold_{}.pkl'.format(self.wavelet, self.hold)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/test_multiclass_hold_{}.pkl'.format(self.wavelet, self.hold)), 'rb') as f:
                self.test_X = pickle.load(f)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)
        self.train_spread_long = self.train_X.pop('spread_long', None)
        self.train_spread_short = self.train_X.pop('spread_short', None)
        self.test_spread_long = self.test_X.pop('spread_long', None)
        self.test_spread_short = self.test_X.pop('spread_short', None)

    def train_test(self, bidirect=False, rec_layer=0, lstm_l2=1e-3, lstm_units=256, lstm_layer=2, lstm_dropout=0.25, lstm_recurrent_dropout=0,
                   dense_l2=1e-3, dense_unit1=128, dense_unit2=0, dense_unit3=0, dense_act_f=0, dense_drop=0.25, BatchNorm=True, batch_size=160, epochs=2000, save_model=False):
        '''
        model create
        '''
        rec_layer = GRU if rec_layer else LSTM
        model_input = []
        model_output = []
        n_lstm = len(self.train_X.keys())
        lstm_reg = l2(lstm_l2)
        if (not self.denoise) and ('pure' not in self.wavelet):
            n_lstm //= 5
        for _key in range(n_lstm):
            x = i = Input(shape=(self.time_step, 1)) if self.denoise or ('pure' in self.wavelet) else Input(shape=(self.time_step, 5))
            for _layer in range(lstm_layer - 1):
                if bidirect:
                    x = Bidirectional(rec_layer(units=lstm_units, kernel_regularizer=lstm_reg,
                                                dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True))(x)
                else:
                    x = rec_layer(units=lstm_units, kernel_regularizer=lstm_reg, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True)(x)
            # Apply a single dense layer to all timesteps of the resulting sequence to convert back to prices
            if bidirect:
                x = Bidirectional(rec_layer(units=lstm_units, kernel_regularizer=lstm_reg, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(x)
            else:
                x = rec_layer(units=lstm_units, kernel_regularizer=lstm_reg, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(x)
            m = Model(i, x)
            model_input += [m.input]
            model_output += [m.output]
        x = Concatenate()(model_output)
        dense_act_f = ['relu', 'selu', 'tanh', 'elu', 'leakyrelu'][dense_act_f]
        dense_reg = l2(dense_l2)
        dense_units = [dense_unit1, dense_unit2, dense_unit3]
        dense_units.sort(reverse=True)
        for dense_unit in dense_units:
            if dense_unit > 0:
                x = Dense(dense_unit, kernel_regularizer=dense_reg)(x)
                if BatchNorm:
                    x = BatchNormalization()(x)
                if dense_act_f == 'leakyrelu':
                    x = LeakyReLU(alpha=0.1)(x)
                else:
                    x = Activation(activation=dense_act_f)(x)
                if dense_drop > 0:
                    x = Dropout(dense_drop)(x)
        # x = Dense(4, activation='softmax')(x)
        x = Dense(3, activation='softmax')(x)
        model = Model(model_input, x)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, clipnorm=1.), metrics=['accuracy', long_short_metric])
        print('{} {} n_layer={}, l2={:.8f}\tDense activation: {}, l2={:.8f}, dropout=[{:.4f}, {:.4f}, {:.4f}]'.format(
            self.wavelet, rec_layer, lstm_layer, lstm_l2, dense_act_f, dense_l2, lstm_dropout, lstm_recurrent_dropout, dense_drop))

        '''
        input data create & normalization
        '''
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        if (not self.denoise) and ('pure' not in self.wavelet):
            inputTrain_X = [[v[1] for v in self.train_X.items() if str(i) in v[0]] for i in range(1, n_lstm + 1)]
            inputTest_X = [[v[1] for v in self.test_X.items() if str(i) in v[0]] for i in range(1, n_lstm + 1)]
            for i in range(len(inputTrain_X)):
                for j in range(5):
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

        inputTrain_Y = to_categorical(self.train_Y)

        '''
        Downsample Majority Class To Match Minority Class
        '''
        resample_size = 80
        cls0_idx = len(self.train_Y) - 500 + np.random.choice(np.where(self.train_Y[-500:] == 0)[0], size=resample_size, replace=False)
        cls1_idx = len(self.train_Y) - 500 + np.random.choice(np.where(self.train_Y[-500:] == 1)[0], size=resample_size, replace=False)
        cls2_idx = len(self.train_Y) - 500 + np.random.choice(np.where(self.train_Y[-500:] == 2)[0], size=resample_size, replace=False)
        # cls3_idx = len(self.train_Y) - 500 + np.random.choice(np.where(self.train_Y[-500:] == 3)[0], size=resample_size, replace=False)
        # down_idx = np.hstack((cls0_idx, cls1_idx, cls2_idx, cls3_idx))
        down_idx = np.hstack((cls0_idx, cls1_idx, cls2_idx))
        np.random.shuffle(down_idx)
        validTest_X = list()
        for i in range(len(inputTrain_X)):
            inputTrain_X[i] = np.array(inputTrain_X[i])
            validTest_X.append(inputTrain_X[i][down_idx])
            inputTrain_X[i] = np.delete(inputTrain_X[i], down_idx, 0)
        inputTrain_Y = np.array(inputTrain_Y)
        validTest_Y = inputTrain_Y[down_idx]
        valid_spread_long = self.train_spread_long[down_idx]
        valid_spread_short = self.train_spread_short[down_idx]
        inputTrain_Y = np.delete(inputTrain_Y, down_idx, 0)
        train_spread_long = np.delete(self.train_spread_long, down_idx, 0)
        train_spread_short = np.delete(self.train_spread_short, down_idx, 0)

        '''
        model train
        '''
        if self.denoise:
            checkpoint_path = './model/checkpoint/for_opt_denoise_multiclass_{}.h5'.format(self.wavelet)

        else:
            checkpoint_path = './model/checkpoint/for_opt_multiclass_{}.h5'.format(self.wavelet)

        # class_weight = {0: 1 - sum(self.train_Y == 0) / len(self.train_Y),
        #                 1: 1 - sum(self.train_Y == 1) / len(self.train_Y),
        #                 2: 1 - sum(self.train_Y == 2) / len(self.train_Y),
        #                 3: 1 - sum(self.train_Y == 3) / len(self.train_Y)}
        class_weight = {0: 1 - sum(self.train_Y == 0) / len(self.train_Y),
                        1: 1 - sum(self.train_Y == 1) / len(self.train_Y),
                        2: 1 - sum(self.train_Y == 2) / len(self.train_Y)}
        get_best_model = CustomCheckpoint_multiclass(checkpoint_path)
        reduce_lr = ReduceLROnPlateau(monitor='long_short_metric', patience=60, factor=0.8, mode='auto', verbose=1)
        hLS = model.fit(inputTrain_X, inputTrain_Y, validation_data=(validTest_X, validTest_Y), class_weight=class_weight,
                        epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr, get_best_model], verbose=2)

        if not get_best_model.valid:
            del model
            K.clear_session()
            return 0

        '''
        trend predict
        '''
        model = load_model(checkpoint_path, custom_objects={'long_short_metric': long_short_metric})
        trainPredict = np.argmax(model.predict(inputTrain_X), 1)
        validPredict = np.argmax(model.predict(validTest_X), 1)
        testPredict = np.argmax(model.predict(inputTest_X), 1)

        '''
        accuracy estimate
        '''
        acc_in = sum(trainPredict == np.argmax(inputTrain_Y, 1)) / len(trainPredict)
        acc_val = sum(validPredict == np.argmax(validTest_Y, 1)) / len(validPredict)
        acc_out = sum(testPredict == self.test_Y) / len(testPredict)
        print('[cuckoo] ----------> \tAcc_in: {:.8f} \tAcc_val: {:.8f} \tAcc_out: {:.8f}\n'.format(acc_in, acc_val, acc_out))

        '''
        return rate estimate
        '''
        rr_in = sum(((trainPredict == 0) * train_spread_long) + (-1 * (trainPredict == 1) * train_spread_short))
        rr_val = sum(((validPredict == 0) * valid_spread_long) + (-1 * (validPredict == 1) * valid_spread_short))
        rr_out = sum(((testPredict == 0) * self.test_spread_long) + (-1 * (testPredict == 1) * self.test_spread_short))
        print('[cuckoo] ----------> \trr_in: {:.8f} \trr_val: {:.8f} \trr_out: {:.8f}\n'.format(rr_in, rr_val, rr_out))

        '''
        plot loss curve & price curve
        '''
        plt.figure()
        epochs = hLS.epoch[-1]
        plt.plot(range(epochs), hLS.history['loss'][:epochs], color='blue', label='accuracy')
        plt.plot(range(epochs), hLS.history['val_loss'][:epochs], color='red', label='val_accuracy')
        plt.xlabel('epoch')
        plt.legend()
        if self.denoise:
            plt.savefig(Path('losscurve/{:.3f}_for_opt_denoise_multiclass_{}.png'.format(acc_out, self.wavelet)))
        else:
            plt.savefig(Path('losscurve/{:.3f}_for_opt_multiclass_{}.png'.format(acc_out, self.wavelet)))

        if self.plot:
            pass

        '''
        save model & return
        '''
        if rr_val > 0:
            save_dir = Path('./model/for_opt/{}'.format(self.wavelet))
            save_dir.mkdir(exist_ok=True)
            save_file = save_dir / '{}.h5'.format(rr_val)
            if save_model:
                model.save(save_file)

            del model
            K.clear_session()

            return rr_val

        return 0


if __name__ == "__main__":
    for w in ['haar_20']:
        lstm = opt_multi_LSTM(wavelet=w, plot=True)
        lstm.train_test()
