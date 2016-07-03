from __future__ import division
import numpy as np


def mean_squared_error(i, j):
    return np.sum(np.power(i - j, 2))


def correlation(i, j):
    return np.corrcoef(i, j)
