from __future__ import division
import numpy as np

def mean_squared_error(x, y):
    return np.sum(np.power(y - x, 2))


def precision(x, y):
    assert x.shape != y.shape, 'Shape of Input data should be same ' \
                               'when you want to get precision.'
    right_counter = 0
    for i, j in zip(x, y):
        i_idx = np.argwhere(i == 1)
        j_idx = np.argwhere(j == 1)
        assert len(i_idx) == 1, 'Only predict class number can be seted as one in Input x'
        assert len(j_idx) == 1, 'Only predict class number can be seted as one in Input Y'
        if j_idx == i_idx:
            right_counter += 1
    return right_counter / len(x)

