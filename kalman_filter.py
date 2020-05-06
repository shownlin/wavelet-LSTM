import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from pykalman import UnscentedKalmanFilter
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
        def get_weight_grad(model, inputs, outputs):
            """ Gets gradient of model for given inputs and outputs for all weights"""
            grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
            symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
            f = K.function(symb_inputs, grads)
            x, y, sample_weight = model._standardize_user_data(inputs, outputs)
            output_grad = f(x + y + sample_weight)
            return output_grad

        def get_layer_output_grad(model, inputs, outputs, layer=-1):
            """ Gets gradient a layer output for given inputs and outputs"""
            grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
            symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
            f = K.function(symb_inputs, grads)
            x, y, sample_weight = model._standardize_user_data(inputs, outputs)
            output_grad = f(x + y + sample_weight)
            return output_grad

        a = get_weight_grad(model, dummy_in, dummy_out)
        b = get_layer_output_grad(model, dummy_in, dummy_out, layer=0)
        c = get_layer_output_grad(model, dummy_in, dummy_out, layer=1)
        print(a)
        # print(b)
        # print()
'''


class Kalman():
    def __init__(self, save_model=False):
        self.save_model = save_model
        self.load_data()

    def load_data(self):
        origin_dir = Path('./OriginData')
        orgin_file = origin_dir / 'TX_price.csv'
        self.df = pd.read_csv(orgin_file, parse_dates=['date'], infer_datetime_format=True)
        self.dates = self.df['date']
        features = ['open', 'high', 'low', 'close', 'volume', 'adj_close']

        with open(Path('./pure_16/train.pkl'), 'rb') as f:
            self.train_X = pickle.load(f)
        with open(Path('./pure_16/test.pkl'), 'rb') as f:
            self.test_X = pickle.load(f)

        self.train_Y = self.train_X.pop('y', None)
        self.test_Y = self.test_X.pop('y', None)
        self.train_date = self.train_X.pop('date', None)
        self.test_date = self.test_X.pop('date', None)
        self.time_step = self.train_X.pop('time_step', None)

    def train(self):

        stock_price = self.df['close'].values
        stock_change = self.df['close'].diff().values
        measurements = np.stack((stock_price, stock_change), axis=-1)[1:]
        model_input = list()
        model_output = list()

        m = Sequential()
        m.add(LSTM(units=10, input_shape=(None, 1)))
        model_input += [m.input]
        model_output += [m.output]

        x = Concatenate()(model_output)
        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)

        @tf.function
        def asdqqq():
            with tf.GradientTape() as g:
                x = tf.cast(dummy_in, tf.float32)
                g.watch(x)
                g.watch(model.trainable_weights)
                result = model(x)
                # model.predict(dummy_in[0])
            return g.batch_jacobian(result, x)

        Q = Sequential()
        Q.add(LSTM(units=10, input_shape=(None, 1)))
        Q.add(Dense(1))


model.add(Reshape((n_steps_out, n_features)))

  @tf.function
   def asdkk():
        with tf.GradientTape() as g:

            x = tf.cast(dummy_in, tf.float32)
            g.watch(x)
            g.watch(model.trainable_weights)
            result = model(x)
            # model.predict(dummy_in[0])
        return g.jacobian(result, x)

    # a = asdkk().numpy()
    b = asdqqq().numpy()
    # print(a)

    print()


if __name__ == "__main__":
    kal = Kalman()
    kal.train()
