from __future__ import division
import numpy as np


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_d(x):
    l = logistic(x)
    return np.array(list(map(lambda x: x * (1 - x), l)))


def linear(x):
    return x


def linear_d(x):
    return np.ones(x.shape)


def softmax(x):
    y = np.exp(x)
    s = np.sum(y)
    return y / s
