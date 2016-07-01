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

    def _computer_delta(self, X, Y):
        """
            computer delta of each net

            According to the theory provided below:
            X is one kind of input data modality, and H1 is transposed matrix of X
            Y is another kind of input data modality, and H2 is transposed matrix of Y

            acknowledge:G. Andrew, R. Arora, J. Bilmes, and K. Livescu. Deep canonical correlation analysis. In ICML, 2013.
        """
        pass
