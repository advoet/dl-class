from .layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.filters = None
        self.height = None
        self.width = None

    def forward(self, data):
        # data comes in as N x 16 x 7 x 7
        # TODO reshape the data here and return it (this can be in place).
        self.batch_size, self.filters, self.height, self.width = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
        return np.reshape(data, (data.shape[0],-1))

    def backward(self, previous_partial_gradient):
        # ppg is N x 16*7*7
        return np.reshape(previous_partial_gradient, 
                            (
                            self.batch_size,
                            self.filters,
                            self.height,
                            self.width
                            ))
