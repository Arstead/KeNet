from __future__ import division
from KeNet.layers.layer import logistic_layer
from KeNet.layers.layer import softmax_layer
from KeNet.net.NeuralNet import *
from KeNet.dataset.mnist import *

__version__ = '0.1'


def fun(x):
    w = np.array([0.1, 0.2, 0.1])
    b = 0.2
    return np.dot(x, w) + b


if __name__ == '__main__':

    m = Mnist()
    (train_data, train_labels), (test_data, test_labels) = m.load(is_to_categorical=True)

    train_data /= 255
    test_data /= 255

    print('__main__')
    nn = NeuralNet()
    nn.add_layer(logistic_layer(nodes_num=784, is_input=True))
    nn.add_layer(logistic_layer(nodes_num=1000))
    nn.add_layer(softmax_layer(nodes_num=10))
    nn.complie(epoch=400, learning_rate=0.1)

    nn.train(train_data[0:100], train_labels[0:100], test_data[0:20], test_labels[0:20])


