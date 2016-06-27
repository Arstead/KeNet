import numpy as np

def mean_squared_error(x,y):
    return np.sum(np.power(y - x, 2))