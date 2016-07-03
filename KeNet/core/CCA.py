from __future__ import division
import numpy as np


def CCA(X, Y, n=None, r=0):
    """
    PARAMETER:
        X is a m * n1 matrix
            m is number of samples, which also means number of observations
            n1 is number of features in modality X, which also means number of variables
        Y is a m * n2 matrix
            m is number of samples, which also means number of observations
            n2 is number of features in modality Y, which also means number of variables
        n is number of components
        r is regularized parameter
    RETURN:
        W_x:
            project weight for X
        W_y:
            project weight for Y
        Corr:
            correlation of projected X and projected Y
        X_:
            projected X
        Y_:
            projected Y
    References:
        http://blog.csdn.net/statdm/article/details/7585113
    """
    m1, n1 = X.shape
    m2, n2 = Y.shape
    assert m1 == m2, 'CCA: Inpute data need to contain same number of samples'
    k = min(n1, n2)
    if n is not None:
        assert n <= k, 'CCA: Number of components must less than min(n1, n2)'
    else:
        n = k

    R = np.c_[X, Y]
    sigma = np.cov(R, rowvar=False)
    sigma_XX = sigma[0:n1, 0:n1] + r * np.eye(n1)
    sigma_XY = sigma[0:n1, n1:n1 + n2]
    sigma_YX = sigma[n1:n1 + n2, 0:n1]
    sigma_YY = sigma[n1:n1 + n2, n1:n1 + n2] + r * np.eye(n2)

    assert np.linalg.det(sigma_XX) == 0.0 or np.linalg.det(
        sigma_XX) == 0.0, 'CCA warning: covariance matrix cov(X,X) or cov(Y,Y) is singular, ' \
                          'please add regularized parameter to adapt those matrix.'
    E1 = np.linalg.inv(sigma_XX).dot(sigma_XY)
    E2 = np.linalg.inv(sigma_YY).dot(sigma_YX)

    A = np.dot(E1, E2)
    A_eig, A_eigvec = np.linalg.eig(A)
    A_order_idx = np.argsort(-A_eig)  # reversed order
    W_x = A_eigvec[:, A_order_idx[0:n]]
    W_y = E2.dot(W_x)

    # regularize weight x
    M = W_x.T.dot(sigma_XX).dot(W_x)
    for i in range(k):
        W_x[:, i] /= np.sqrt(M[i, i])
    # regularize weight x
    M = W_y.T.dot(sigma_YY).dot(W_y)
    for i in range(k):
        W_y[:, i] /= np.sqrt(M[i, i])

    X_ = X.dot(W_x)
    Y_ = Y.dot(W_y)
    Corr = np.sqrt(A_eig[A_order_idx[0:n]])

    return W_x, W_y, Corr, X_, Y_


def cov(X, Y=None, r=0.0):
    """
    PARAMETER:
        X is a n1 * m matrix
            n1 is number of features in modality X, which also means number of variables
            m is number of samples, which also means number of observations
        Y is a n2 * m matrix
            n2 is number of features in modality Y, which also means number of variables
            m is number of samples, which also means number of observations
        r is regularization parameter
    RETURN:
        sigma:
            covariance of X and Y
    FURTHER IMPROVE:
        1. this function don't implement computer covariance along axis 0
    """
    if X.ndim == 1:
        X = X.reshape((1, -1))

    n1, m1 = X.shape
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_centered = X - X_mean
    if Y is not None:
        if Y.ndim == 1:
            Y = Y.reshape((1, -1))
        n2, m2 = Y.shape
        assert m1 == m2, 'cov: Inpute data need to contain same number of samples'
        Y_mean = np.mean(Y, axis=1).reshape(-1, 1)
        Y_centered = Y - Y_mean
        R = np.r_[X_centered, Y_centered]
        I = np.ones((n1 + n2, n1 + n2))
    else:
        I = np.ones((n1, n1))
        R = X_centered

    sigma = 1 / (m1 - 1) * np.dot(R, R.T) + r * I
    return sigma


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 6], [4, 2], [5, 2], [1, 2]])
    # A = np.array([1, 2, 2, 3, 4])
    B = np.array([9, 7, 12, 15, 17])
    print(cov(A.T, B))
    print(np.cov(A.T, B))
