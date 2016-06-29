from __future__ import division
import numpy as np

def mean_squared_error(A, B):
    assert A.shape != B.shape, 'Shape of Input data should be same ' \
                               'when you want to get precision.'
    error_counter = 0.0
    for i, j in zip(A, B):
        error_counter += np.sum(np.power(i - j, 2))
    return error_counter


def precision(A, B):
    assert A.shape != B.shape, 'Shape of Input data should be same ' \
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

