import numpy as np
from ..core import activation_function as AF


class layer(object):
    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        super(layer, self).__init__(*args, **kwargs)
        self.is_input = is_input
        self.nodes_num = nodes_num
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        self.delta = None

    def activation(self):
        pass

    def activation_d(self):
        pass

    def build(self, input_shape):
        if not self.is_input:
            self.weights = 0.05 * np.random.random([input_shape, self.nodes_num])
            self.bias = 0.05 * np.random.random(self.nodes_num)


class logistic_layer(layer):
    """
        logistic layer
    """

    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        super(logistic_layer, self).__init__(nodes_num, is_input, *args, **kwargs)

    def activation(self, x=None):
        return AF.logistic(self.input) if input is None else AF.logistic(x)

    def activation_d(self, x=None):
        return AF.logistic_d(self.input) if input is None else AF.logistic_d(x)


class linear_layer(layer):
    """
        linear layer
    """

    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        super(linear_layer, self).__init__(nodes_num, is_input, *args, **kwargs)

    def activation(self, x=None):
        return AF.linear(self.input) if input is None else AF.logistic(x)

    def activation_d(self, x=None):
        return AF.linear_d(self.input) if input is None else AF.logistic_d(x)


class softmax_layer(layer):
    """
        softmax layer
    """

    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        super(softmax_layer, self).__init__(nodes_num, is_input, *args, **kwargs)

    def activation(self, x=None):
        return AF.softmax(self.input) if input is None else AF.logistic(x)

    def activation_d(self, x=None):
        raise Exception('softmax layer only can be output layer!')
