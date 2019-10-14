import numpy as np
from numba import njit, prange

from .layer import Layer
from nn import Parameter


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.data = None
        self.initialize()

    def forward(self, data):
        '''
        ReLU Layer forward pass

        :param data: n x d (batch x features)
        :return: n x d array relu activated (batch x channels) 
        '''
        self.data = Parameter(data)
        return np.clip(data, 0, None)

    def backward(self, previous_partial_gradient):
        '''
        ReLU Layer backward pass

        :param previous_partial_gradient: n x c of gradients w.r.t. future layer
        :return: gradients w.r.t. inputs
        '''
        return np.where(self.data.data > 0, previous_partial_gradient, 0)

class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        pass

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        pass

    def backward(self, previous_partial_gradient):
        # TODO
        self.backward_numba(self.data, previous_partial_gradient)
        return None
