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
from lib.custom_callback import CustomCheckpoint_teacher
from lib.attention_withcontext import AttentionWithContext
from lib.attention2 import Attention
from lib.attention3 import AttentionWeightedAverage
from lib.attention import attention_3d_block
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


class techer_DNN():
    def __init__(self, denoise=None, wavelet='pure_16'):
        self.wavelet = wavelet
        self.denoise = denoise
        self.load_data()

    def load_data(self):
        if self.denoise:
            with open(Path('./{}/denoise/train_multiclass.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/denoise/test_multiclass.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)
        else:
            with open(Path('./{}/train_multiclass.pkl'.format(self.wavelet)), 'rb') as f:
                self.train_X = pickle.load(f)
            with open(Path('./{}/test_multiclass.pkl'.format(self.wavelet)), 'rb') as f:
                self.test_X = pickle.load(f)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)

    def train_test(self, batch_size=160, epochs=2000, save_model=False):
        '''
        model create
        '''
        model_input = []
        model_output = []
        dense_reg = l2(1e-2)
        n_lstm = len(self.train_X.keys())
        if (not self.denoise) and ('pure' not in self.wavelet):
            n_lstm //= 6
        for _key in range(n_lstm):
            x = i = Input(shape=(self.time_step, 1)) if self.denoise or ('pure' in self.wavelet) else Input(shape=(self.time_step, 6))
            x = Dense(128, kernel_regularizer=dense_reg)(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, kernel_regularizer=dense_reg)(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, kernel_regularizer=dense_reg)(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = Dropout(0.3)(x)
            m = Model(i, x)
            model_input += [m.input]
            model_output += [m.output]

        x = Concatenate()(model_output)
        x = Flatten()(x)
        x = Dense(512, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, kernel_regularizer=dense_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(4, activation='softmax')(x)
        model = Model(model_input, x)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, clipnorm=1.), metrics=['accuracy', long_short_metric])

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

        inputTrain_Y = to_categorical(self.train_Y)
        inputTest_Y = to_categorical(self.test_Y)

        '''
        Downsample Majority Class To Match Minority Class
        '''
        resample_size = 120
        cls0_idx = np.random.choice(np.where(self.train_Y == 0)[0], size=resample_size, replace=False)
        cls1_idx = np.random.choice(np.where(self.train_Y == 1)[0], size=resample_size, replace=False)
        cls2_idx = np.random.choice(np.where(self.train_Y == 2)[0], size=resample_size, replace=False)
        cls3_idx = np.random.choice(np.where(self.train_Y == 3)[0], size=resample_size, replace=False)
        down_idx = np.hstack((cls0_idx, cls1_idx, cls2_idx, cls3_idx))
        np.random.shuffle(down_idx)
        validTest_X = list()
        for i in range(len(inputTrain_X)):
            inputTrain_X[i] = np.array(inputTrain_X[i])
            validTest_X.append(inputTrain_X[i][down_idx])
            inputTrain_X[i] = np.delete(inputTrain_X[i], down_idx, 0)
        inputTrain_Y = np.array(inputTrain_Y)
        validTest_Y = inputTrain_Y[down_idx]
        inputTrain_Y = np.delete(inputTrain_Y, down_idx, 0)

        '''
        model train
        '''
        if self.denoise:
            checkpoint_path = './model/checkpoint/teacher_{}.h5'.format(self.wavelet)

        else:
            checkpoint_path = './model/checkpoint/teacher_{}.h5'.format(self.wavelet)

        class_weight = {0: 1 - sum(self.train_Y == 0) / len(self.train_Y),
                        1: 1 - sum(self.train_Y == 1) / len(self.train_Y),
                        2: 1 - sum(self.train_Y == 2) / len(self.train_Y),
                        3: 1 - sum(self.train_Y == 3) / len(self.train_Y)}
        get_best_model = CustomCheckpoint_teacher(checkpoint_path)
        reduce_lr = ReduceLROnPlateau(monitor='accuracy', patience=30, factor=0.8, mode='auto', verbose=1)
        hLS = model.fit(inputTrain_X, inputTrain_Y, validation_data=(validTest_X, validTest_Y), class_weight=class_weight,
                        epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr, get_best_model], verbose=2)
        print('best epoch = {}, train best={}, best_val_long_short_metric={}'.format(get_best_model.best_epoch, get_best_model.train_best, get_best_model.best))

        if not get_best_model.valid:
            del model
            K.clear_session()
            return 0

        '''
        trend predict
        '''
        model = load_model(checkpoint_path, custom_objects={'long_short_metric': long_short_metric})
        trainPredict = np.argmax(model.predict(inputTrain_X), 1)
        testPredict = np.argmax(model.predict(inputTest_X), 1)

        '''
        accuracy estimate
        '''
        acc_in = sum(trainPredict == np.argmax(inputTrain_Y, 1)) / len(trainPredict)
        acc_out = sum(testPredict == np.argmax(inputTest_Y, 1)) / len(testPredict)
        print('[cuckoo] ----------> \tAcc_in: {:.8f} \tAcc_out: {:.8f}\n'.format(acc_in, acc_out))

        '''
        save model
        '''
        save_dir = Path('./model/for_opt/teacher_{}'.format(self.wavelet))
        save_dir.mkdir(exist_ok=True)
        save_file = save_dir / '{}.h5'.format(acc_out)
        if save_model:
            model.save(save_file)

        del model
        K.clear_session()

        return acc_out


if __name__ == "__main__":
    for w in ['haar_16']:
        lstm = techer_DNN(wavelet=w)
        lstm.train_test()
