import numpy as np
from ..core import activation_function as AF


class layer(object):
    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        self.is_input = is_input
        self.nodes_num = nodes_num
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        self.delta = None
        return super(layer, self).__init__(*args, **kwargs)

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
        return super(logistic_layer, self).__init__(nodes_num, is_input, *args, **kwargs)

    def activation(self, input=None):
        return AF.logistic(self.input) if input is None else AF.logistic(input)

    def activation_d(self, input=None):
        return AF.logistic_d(self.input) if input is None else AF.logistic_d(input)


class linear_layer(layer):
    """
        linear layer
    """
    def __init__(self, nodes_num: 'number of nodes', is_input: 'a flag to tell is it an input layer' = False, *args,
                 **kwargs):
        return super(linear_layer, self).__init__(nodes_num, is_input, *args, **kwargs)


    def activation(self, input=None):
        return AF.linear(self.input) if input is None else AF.logistic(input)


    def activation_d(self, input=None):
        return AF.linear_d(self.input) if input is None else AF.logistic_d(input)
