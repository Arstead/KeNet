from KeNet.layers.layer import logistic_layer
from KeNet.layers.layer import linear_layer
from KeNet.net.NeuralNet import *

__version__ = '0.1'


def fun(x):
    w = np.array([0.1, 0.2, 0.1])
    b = 0.2
    return np.dot(x, w) + b


if __name__ == '__main__':
    print('__main__')
    nn = NeuralNet()
    nn.add_layer(linear_layer(nodes_num=3, is_input=True))
    nn.add_layer(linear_layer(nodes_num=4))
    nn.add_layer(linear_layer(nodes_num=1))
    nn.complie(epoch=300, learning_rate=0.1)

    x = np.random.random([100, 3])
    z = fun(x)
    y = fun(x) + 0.02 * np.random.random([100])

    x_t = np.random.random([20, 3])
    y_t = fun(x_t) + 0.02 * np.random.random([20])

    nn.train(x, y, x_t, y_t)

    o = nn.predict(x)

    print(np.sum((o.reshape(1, -1) - y)**2))
