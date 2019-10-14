import numpy as np
from numba import njit, prange

from .layer import Layer
from nn import Parameter

class LeakyReLULayer(Layer):
    def __init__(self, slope: float = .9, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.data = None
        self.initialize()

    def forward(self, data):
        self.data = Parameter(data)
        return np.where(data > 0, data, self.slope*data)

    def backward(self, previous_partial_gradient):
        return np.where(self.data.data > 0, previous_partial_gradient, self.slope*previous_partial_gradient)
