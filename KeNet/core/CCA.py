from __future__ import division
import numpy as np


def CCA(X, Y):
    pass

def cov(X, Y, r:'regularization parameter' = 0):
    """
        X is a n1 * m matrix
            n1 is number of features in modality X
            m is number of samples
        Y is a n2 * m matrix
            n2 is number of features in modality Y
            m is number of samples
    """
    if X.ndim == 1:
        X = X.reshape((1, -1))
    if Y.ndim == 1:
        Y = Y.reshape((1, -1))

    n1, m1 = X.shape
    n2, m2 = Y.shape
    assert m1 == m2, 'covariances: Inpute data need to contain same number of samples'
    X_mean = np.mean(X, axis=1)
    Y_mean = np.mean(Y, axis=1)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    sigma = 1 / (m1) * np.dot(X_centered, Y_centered.T) + r * np.ones((n1, n2))
    return sigma

if __name__ == '__main__':
    A = np.array([[3, 2, 4, 5, 6], [1, 2, 3, 4, 6]])
    B = np.array([9, 7, 12, 15, 17])
    print(cov(A, B))
    print(np.cov(A, B))


