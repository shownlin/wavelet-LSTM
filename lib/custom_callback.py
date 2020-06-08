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
        self.stop_criti = -np.Inf
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
            if np.greater(current, self.stop_criti):
                self.stop_criti = current
            if self.save_best_only:
                if np.greater(self.train_best, self.baseline * 1.05) and np.greater(self.train_best, current) and np.greater(current, self.best):
                    self.best = current
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
        elif np.equal(epoch, 200) and np.less(self.stop_criti, self.baseline):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 250) and np.less(self.train_best, self.baseline * 1.05):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 400) and np.less(self.best, self.baseline * 1.05):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 600) and np.greater(self.train_best, 0.8) and np.greater(self.best, self.baseline):
            self.model.stop_training = True
            print('Early finish at %d' % epoch)


class CustomCheckpoint_multiclass(Callback):

    def __init__(self, filepath, save_best_only=True, save_weights_only=False, period=1):
        super(CustomCheckpoint_multiclass, self).__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.stop_criti = -np.Inf
        self.best = -np.Inf
        self.best_epoch = 0
        # self.train_best = -np.Inf
        self.valid = False
        self.train_acc = list()
        self.train_metric = list()
        self.val_acc = list()
        self.val_metric = list()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            # train_acc = logs.get('long_short_metric')
            # if np.greater(train_acc, self.train_best):
            #     self.train_best = train_acc

            self.train_acc.append(logs.get('accuracy'))
            self.train_metric.append(logs.get('long_short_metric'))
            self.train_acc, self.train_metric = self.train_acc[-10:], self.train_metric[-10:]
            train_acc_mean, train_metric_mean = np.mean(self.train_acc), np.mean(self.train_metric)

            self.val_acc.append(logs.get('val_accuracy'))
            self.val_metric.append(logs.get('val_long_short_metric'))
            self.val_acc, self.val_metric = self.val_acc[-10:], self.val_metric[-10:]
            val_acc_mean, val_metric_mean = np.mean(self.val_acc), np.mean(self.val_metric)

            if np.greater(self.val_metric[-1], self.stop_criti):
                self.stop_criti = self.val_metric[-1]
            if self.save_best_only:
                if np.greater(train_acc_mean, 0.5) and np.greater(self.val_metric[-1], 0.5) and np.greater(self.val_metric[-1], self.best):
                    self.best = self.val_metric[-1]
                    self.best_epoch = epoch
                    if self.val_metric[-1] is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % ('val_long_short_metric'), RuntimeWarning)
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
        if np.equal(epoch, 200) and np.less(val_acc_mean, 0.37):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 250) and np.less(self.stop_criti, 0.5375):
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.equal(epoch, 500) and not self.valid:
            self.model.stop_training = True
            print('Early Stop at %d' % epoch)
        elif np.greater(epoch, 300) and np.greater(train_acc_mean - self.best, 0.3):
            self.model.stop_training = True
            print('Early finish at %d' % epoch)
        elif np.greater(train_acc_mean, 0.8) and np.greater(self.best, 0.6):
            self.model.stop_training = True
            print('Early finish at %d' % epoch)


class CustomCheckpoint_teacher(Callback):

    def __init__(self, filepath, save_best_only=True, save_weights_only=False,
                 period=1):
        super(CustomCheckpoint_teacher, self).__init__()
        self.monitor = 'val_long_short_metric'
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best = -np.Inf
        self.train_best = -np.Inf
        self.valid = False
        self.best_epoch = 0

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

            if self.save_best_only:
                if np.greater(train_acc, 0.5) and np.greater(current, self.best):
                    self.best = current
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            self.valid = True
                            self.best_epoch = epoch
                        else:
                            self.model.save(filepath, overwrite=True)
                            self.valid = True
                            self.best_epoch = epoch
            else:
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    self.valid = True
                else:
                    self.model.save(filepath, overwrite=True)
                    self.valid = True
        if np.equal(epoch, 300) and np.greater(self.train_best, 0.8) and np.greater(self.best, 0.55):
            self.model.stop_training = True
            print('Early finish at %d' % epoch)
