# -*- coding:UTF-8 -*-

from ..layers import *
from .Net import *
from ..core import error_function as EF
from ..dataset.utilor import to_labels
import numpy as np


class NeuralNet(Net):
    def __init__(self, *args, **kwargs):
        super(NeuralNet, self).__init__(*args, **kwargs)
        self._is_back_propagation = False
        self._is_feedforward = False

    def _feedforward(self, X):
        assert X.ndim == 2, '_feedforward:Input data must be an M x N matrix. ' \
                            'M is num of samples, N is num of features for each sample'
        Y = np.zeros([len(X), self.layers[-1].nodes_num])

        assert self.is_complie, 'complie network before you use it.'
        self.layers[0].output = X
        for layer_idx in range(1, len(self.layers)):
            self.layers[layer_idx].input = np.dot(self.layers[layer_idx - 1].output, self.layers[layer_idx].weights) + \
                                            self.layers[layer_idx].bias
            self.layers[layer_idx].output = self.layers[layer_idx].activation(self.layers[layer_idx].input)

        Y = self.layers[-1].output
        self._is_feedforward = True
        return Y

    def _back_propagation(self, X, Y, delta=None):

        assert type(X) is np.ndarray, 'input of data x need to be type of ndarry'
        assert type(X) is np.ndarray, 'labels data y need to be type of ndarry'
        if delta:
            assert type(delta) is np.ndarray, 'data delta need to be type of ndarry'
            self._feedforward(X)
        else:
            delta = Y - self._feedforward(X)
        self.layers[-1].delta = delta
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            self.layers[layer_idx].delta = np.dot(self.layers[layer_idx + 1].weights,
                                                  self.layers[layer_idx + 1].delta.T).T * \
                                           self.layers[layer_idx].activation_d(self.layers[layer_idx].input)
        self._is_back_propagation = True

    def _update_weight(self):
        assert self._is_back_propagation, 'do back propagation before update weight'
        samples_num = len(self.layers[0].output)
        for layer_idx in range(1, len(self.layers)):
            self.layers[layer_idx].weights += (1 / samples_num) * self.learning_rate * np.dot(
                self.layers[layer_idx - 1].output.T,
                self.layers[layer_idx].delta
            )
            self.layers[layer_idx].bias += (1 / samples_num) * self.learning_rate * np.sum(self.layers[layer_idx].delta, axis=0)
        self._is_back_propagation = False
        self._is_feedforward = False

    def _computer_error(self, A, B):
        assert A.shape == B.shape, 'Shape of Input data should be same ' \
                                   'when you want to get precision.'

        error_counter = 0.0
        for i, j in zip(A, B):
            error_counter += EF.mean_squared_error(i, j)
        return error_counter / len(A)

    def _computer_precision(self, A, B):
        assert A.shape == B.shape, 'Shape of Input data should be same ' \
                                   'when you want to get precision.'
        right_counter = 0
        for i, j in zip(A, B):
            i_idx = np.argwhere(i == 1)
            j_idx = np.argwhere(j == 1)
            assert len(i_idx) == 1, 'Only predict class number can be seted as one in Input x'
            assert len(j_idx) == 1, 'Only predict class number can be seted as one in Input Y'
            if j_idx == i_idx:
                right_counter += 1
        return right_counter / len(A)

    def _set_class(self, X):
        Y = np.zeros(X.shape)
        for idx, x in enumerate(X):
            Y[idx, np.argmax(x)] = 1
        return Y

    def predict(self, X):
        return self._feedforward(X)

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

        train_samples_num = train_data.shape[0]
        for i in range(self.epoch):
            print('epoch %d:' % i, end='  ')

            # update weight
            batch_num = train_samples_num // self.batch_size
            for batch_idx in range(batch_num):
                T = train_data[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                L = train_labels[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                self._back_propagation(T, L)
                self._update_weight()

            # compute train MSE
            O = self._feedforward(train_data)
            MSE = self._computer_error(O, train_labels)
            print('train mse : %0.4f' % MSE, end='  ')

            # compute train precision
            O = self._feedforward(train_data)
            C = self._set_class(O)
            acc = self._computer_precision(C, train_labels)
            print('train acc : %0.4f' % acc, end='  ')

            if validation_data is not None:
                # compute validation MSE
                O = self._feedforward(validation_data)
                MSE = self._computer_error(O, validation_labels)
                print('validation mse : %0.4f' % MSE, end='  ')

                # compute validation precision
                O = self._feedforward(validation_data)
                C = self._set_class(O)
                acc = self._computer_precision(C, validation_labels)
                print('train acc : %0.4f' % acc, end='  ')
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
