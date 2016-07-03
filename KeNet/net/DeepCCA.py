# -*- coding:UTF-8 -*-

from ..layers import *
from .Net import *

from ..core import error_function as EF
from ..dataset.utilor import to_labels
from ..core.CCA import *
import numpy as np
import functools

class DeepCCA(Net):
    def __init__(self, Net_X, Net_Y, *args, **kwargs):
        super(DeepCCA, self).__init__(*args, **kwargs)
        self.Net_X = Net_X
        self.Net_Y = Net_Y

    def train(self, train_X, train_Y, validation_X=None, validation_Y=None):
        assert type(train_X) is np.ndarray, 'DCCA train: Input of train data X need to be type of ndarry'
        assert type(train_Y) is np.ndarray, 'DCCA train: Input of train data X need to be type of ndarry'
        assert (validation_X is not None) == (validation_Y is not None), \
            'validation set need to carry both X and Y'
        if validation_X is not None:
            assert type(validation_X) is np.ndarray, 'DCCA train: Input of validation data X need to be type of ndarry'
            assert type(validation_Y) is np.ndarray, 'DCCA train: Input of validation data X need to be type of ndarry'


        for i in range(self.epoch):
            print('DCCA: epoch %d:' % i)
            X_output = self.Net_X.predict(train_X)
            Y_output = self.Net_Y.predict(train_Y)
            _, _, _, X_H, Y_H = CCA(X_output, Y_output)
            X_delta = self._computer_delta(X_H, Y_H)
            Y_delta = self._computer_delta(Y_H, X_H)
            self.Net_X.train(train_X, delta=X_delta)
            self.Net_Y.train(train_Y, delta=Y_delta)

            # compute total correlation
            XO = self.Net_X.predict(train_X)
            YO = self.Net_Y.predict(train_Y)
            _, _, _, X_HO, Y_HO = CCA(XO, YO)
            Corr = self._compute_total_correlation(X_HO, Y_HO)
            print('train Corr : %0.4f' % Corr, end='  ')

            if validation_X is not None:
                # compute total correlation
                XO = self.Net_X.predict(validation_X)
                YO = self.Net_Y.predict(validation_Y)
                _, _, _, X_HO, Y_HO = CCA(XO, YO)
                Corr = self._compute_total_correlation(X_HO, Y_HO)
                print('validation Corr : %0.4f' % Corr, end='  ')
            print('')


    def complie(self, learning_rate=0.01, batch_size=1, epoch=1):
        self.Net_X.complie(learning_rate, batch_size, 1)
        self.Net_Y.complie(learning_rate, batch_size, 1)
        self.epoch = epoch

    def predict_X(self, X):
        return self.Net_X.predict(X)

    def predict_Y(self, Y):
        return self.Net_Y.predict(Y)

    def _compute_total_correlation(self, X, Y):
        assert A.shape == B.shape, 'Shape of Input data should be same ' \
                                   'when you want to get total correlation.'

        correlation_counter = 0.0
        for i, j in zip(A.T, B.T):
            correlation_counter += EF.correlation(i, j)
        return correlation_counter / len(A.T)

    def _computer_delta(self, X, Y, r=0.0):
        """
            computer delta of each net, dcorr(X, Y) / dX

        PARAMETER:
            X, m * n1 matrix, is one kind of input data modality, and H1 is transposed matrix of X
                m is number of samples, which also means number of observations
                n1 is number of features in modality X, which also means number of variables
            Y, m * n2 matrix, is another kind of input data modality, and H2 is transposed matrix of Y
                m is number of samples, which also means number of observations
                n2 is number of features in modality Y, which also means number of variables
            r is regularized parameter
        References:
                G. Andrew, R. Arora, J. Bilmes, and K. Livescu. Deep canonical correlation analysis. In ICML, 2013.
        """
        m1, n1 = X.shape
        m2, n2 = Y.shape
        assert m1 == m2, 'DCCA computer delta: Input data need to contain same number of samples'

        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(X, axis=0)
        H_1 = (X - X_mean).T        # H1 bar
        H_2 = (Y - Y_mean).T        # H2 bar

        sigma11 = (1 / m1 - 1) * H_1.dot(H_1.T) + r * np.eye(n1)
        sigma12 = (1 / m1 - 1) * H_1.dot(H_2.T)
        sigma21 = (1 / m1 - 1) * H_2.dot(H_1.T)
        sigma22 = (1 / m1 - 1) * H_2.dot(H_2.T) + r * np.eye(n2)

        assert np.linalg.det(sigma11) != 0, 'DCCA computer delta: sigma11 is singular, ' \
                                            'please use regularized parameter to adapt this matrix'
        assert np.linalg.det(sigma22) != 0, 'DCCA computer delta: sigma22 is singular, ' \
                                            'please use regularized parameter to adapt this matrix'
        T1 = np.inv(np.sqrt(sigma11))
        T3 = np.inv(np.sqrt(sigma22))
        T = T1.dot(sigma12).dot(T3)         # T1 dot sigma12 dot T3
        U, D, V = np.linalg.svd(T)
        delta12 = functools.reduce(lambda x, y: np.dot(x, y), [T1, U, V, T3])
        delta11 = -0.5 * functools.reduce(lambda x, y: np.dot(x, y), [T1, U, D, U.T, T1])

        dcorr = (1 / m1 - 1) * (2 * np.dot(delta11, H_1) + np.dot(delta12, H_2))
        return dcorr

