from KeNet.layers import *

class Net(object):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.layers = []
        self.is_complie = False
        self.learning_rate = None
        self.batch_size = None
        self.epoch = None

    def _feedforward(self, x):
        pass

    def _back_propagation(self, z, y):
        pass

    def _update_weight(self):
        pass

    def predict(self, x):
        pass

    def train(self, train_data, train_labels):
        pass

    def add_layer(self, layer):
        pass