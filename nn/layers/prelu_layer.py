import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None
        self.initialize()

    def forward(self, data):
        self.data = Parameter(data)
        return np.where(data > 0, data, self.slope.data*data)

    def backward(self, previous_partial_gradient):
    	# Need to take the average of the biases instead of sum to avoid overflow
        self.slope.grad = np.mean(np.where(self.data.data > 0, 0, self.data.data), axis = 0)
        print(self.slope.data)
        return np.where(self.data.data > 0, previous_partial_gradient, self.slope.data*previous_partial_gradient)