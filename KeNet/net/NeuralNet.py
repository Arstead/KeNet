# -*- coding:UTF-8 -*-

from ..layers import *
from .Net import *
from ..core import error_function as EF
import numpy as np


class NeuralNet(Net):
    def __init__(self, *args, **kwargs):
        self._is_back_propagation = False
        self._is_feedforward = False
        return super(NeuralNet, self).__init__(*args, **kwargs)

    def _feedforward(self, x):
        assert self.is_complie, 'complie network before you use it.'
        self.layers[0].output = x
        for layer_idx in range(1, len(self.layers)):
            self.layers[layer_idx].input = np.dot(self.layers[layer_idx - 1].output, self.layers[layer_idx].weights) + \
                                           self.layers[layer_idx].bias
            self.layers[layer_idx].output = self.layers[layer_idx].activation(self.layers[layer_idx].input)
        self._is_feedforward = True
        return self.layers[-1].output

    def _back_propagation(self, x, y, delta=None):

        assert type(x) is np.ndarray, 'input of data x need to be type of ndarry'
        assert type(x) is np.ndarray, 'labels data y need to be type of ndarry'
        if delta:
            assert type(delta) is np.ndarray, 'data delta need to be type of ndarry'
            self._feedforward(x)
        else:
            delta = y - self._feedforward(x)
        self.layers[-1].delta = delta
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            self.layers[layer_idx].delta = np.dot(self.layers[layer_idx + 1].weights,
                                                  self.layers[layer_idx + 1].delta.T).T
        self._is_back_propagation = True

    def _update_weight(self):
        assert self._is_back_propagation, 'do back propagation before update weight'
        for layer_idx in range(1, len(self.layers)):
            self.layers[layer_idx].weights = self.layers[layer_idx].weights + \
                                             self.learning_rate * np.outer(
                                                 self.layers[layer_idx - 1].output,
                                                 self.layers[layer_idx].delta *
                                                 self.layers[layer_idx].activation_d(self.layers[layer_idx].input)
                                             )
        self._is_back_propagation = False
        self._is_feedforward = False

    def _computer_error(self, x, y):
        return EF.mean_squared_error(x, y)

    def predict(self, x):
        return self._feedforward(x)

    def train(self, train_data, train_labels, validation_data=None, validation_labels=None):
        assert type(train_data) is np.ndarray, 'input of train data need to be type of ndarry'
        assert type(train_labels) is np.ndarray, 'train data labels need to be type of ndarry'
        assert train_data.shape[0] == train_labels.shape[
            0], 'number of train data must be equal to number of train labels'

        assert (validation_labels is not None) == (validation_data is not None), \
            'validation set need to carry both data and labels'

        if validation_data is not None:
            assert type(validation_data) is np.ndarray, 'input of validation data need to be type of ndarry'
            assert type(validation_labels) is np.ndarray, 'validation data labels need to be type of ndarry'
            assert validation_data.shape[0] == validation_labels.shape[
                0], 'number of validation data must be equal to number of train labels'

        train_sample_num = train_data.shape[0]
        for i in range(self.epoch):
            print('epoch %d:' % i, end='  ')
            # update weight
            for sample_idx in range(train_sample_num):
                self._back_propagation(train_data[sample_idx], train_labels[sample_idx])
                self._update_weight()

            sum_train_error = 0.0
            # compute error
            for sample_idx in range(train_sample_num):
                output = self._feedforward(train_data[sample_idx])
                sum_train_error = sum_train_error + self._computer_error(output, train_labels[sample_idx])
            print('train error : %0.2f' % sum_train_error, end='  ')

            # compute validation error
            if validation_data is not None:
                validation_sample_num = validation_data.shape[0]
                sum_validation_error = 0.0
                for sample_idx in range(validation_sample_num):
                    output = self._feedforward(validation_data[sample_idx])
                    sum_validation_error = sum_validation_error + \
                                           self._computer_error(output, validation_labels[sample_idx])
                print('validation error : %0.2f' % sum_validation_error, end='  ')
            print('')

    def add_layer(self, layer):
        self.layers.append(layer)
        self.is_complie = False

    def complie(self, learning_rate=0.01, batch_size=1, epoch=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        for layer_idx in range(1, len(self.layers)):
            self.layers[layer_idx].build(self.layers[layer_idx - 1].nodes_num)
        self.is_complie = True
