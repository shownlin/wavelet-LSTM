import warnings
import numpy as np
from keras.callbacks import Callback


class CustomCheckpoint(Callback):

    def __init__(self, filepath, baseline,
                 save_best_only=True, save_weights_only=False,
                 period=1):
        super(CustomCheckpoint, self).__init__()
        self.monitor = 'val_accuracy'
        self.filepath = filepath
        self.baseline = baseline
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best = -np.Inf
        self.train_best = -np.Inf
        self.valid = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            train_acc = logs.get('accuracy')
            if np.greater(train_acc, self.train_best):
                self.train_best = train_acc
            current = logs.get(self.monitor)
            if np.greater(current, self.best):
                self.best = current
            if np.greater(self.train_best, self.baseline * 1.05):
                if self.save_best_only:
                    if np.greater(self.train_best, current):
                        if current is None:
                            warnings.warn('Can save best model only with %s available, '
                                          'skipping.' % (self.monitor), RuntimeWarning)
                        else:
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                                self.valid = True
                            else:
                                self.model.save(filepath, overwrite=True)
                                self.valid = True
                else:
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                        self.valid = True
                    else:
                        self.model.save(filepath, overwrite=True)
                        self.valid = True
        if np.equal(epoch, 100) and np.less(self.train_best, self.baseline):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 200) and np.less(self.best, self.baseline):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 250) and np.less(self.train_best, self.baseline * 1.05):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 400) and np.less(self.best, self.baseline):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 600) and np.greater(self.train_best, 0.8) and np.greater(self.best, self.baseline):
            self.model.stop_training = True
            print('Early finish at %d' % epoch)
