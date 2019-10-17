import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # data is N x C x H x W
        # TODO
        return None

    def forward(self, data):
        # TODO
        return None

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data):
        # data is N x C x H x W
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
