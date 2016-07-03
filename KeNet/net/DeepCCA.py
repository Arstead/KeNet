# -*- coding:UTF-8 -*-

from ..layers import *
from .Net import *
from ..core import error_function as EF
from ..dataset.utilor import to_labels
import numpy as np


class DeepCCA(Net):
    def __init__(self, Net_X, Net_Y, *args, **kwargs):
        super(DeepCCA, self).__init__(*args, **kwargs)
        self.Net_X = Net_X
        self.Net_Y = Net_Y

    def train(self, train_data, train_labels, validation_data=None, validation_labels=None):
        pass

    def complie(self, learning_rate=0.01, batch_size=1, epoch=1):
        pass

    def predict_X(self, X):
        return self.Net_X.predict(X)

    def predict_Y(self, Y):
        return self.Net_Y.predict(Y)

    def _computer_delta(self, X, Y, r=0.0):
        """
            computer delta of each net

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
        assert m1 == m2, 'DCCA computer delta: Inpute data need to contain same number of samples'

        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(X, axis=0)
        H_1 = (X - X_mean).T        # H1 bar
        H_2 = (Y - Y_mean).T        # H2 bar

        sigma11 = (1 / m1 - 1) * H_1.dot(H_1.T) + r * np.eye(n1)
        sigma12 = (1 / m1 - 1) * H_1.dot(H_2.T)
        sigma21 = (1 / m1 - 1) * H_2.dot(H_1.T)
        sigma22 = (1 / m1 - 1) * H_2.dot(H_2.T) + r * np.eye(n2)

        pass
