from typing import Optional, Callable

import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(LinearLayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))
        self.weight = Parameter(np.ones((input_size, output_size),dtype=np.float32))
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.initialize()

    def forward(self, data: np.ndarray): # -> np.ndarray:
        '''
        Linear layer (fully connected) forward pass
        :param data: n X d array (batch x features)
        :return: n X c array (batch x channels)
        '''
        # TODO do the linear layer
        self.input = Parameter(data)
        return np.matmul(data, self.weight.data) + self.bias.data

    def backward(self, previous_partial_gradient: np.ndarray): #-> np.ndarray
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future layer
        :return: gradients wrt inputs
        """
        # dL/db = dL/dy but how to deal with batches??? ANS: SUM FOR SOME REASON
        # dL/dw_ij = dL/dyj dy/dwij = dL/dyj * xi
        self.bias.grad = np.sum(previous_partial_gradient, axis = 0)

        batch_w_grad = np.zeros((self.input_size, self.output_size, np.size(previous_partial_gradient, 0)))
        for i in range(0, self.input_size):
            for j in range(0, self.output_size):
                batch_w_grad[i,j,:] = np.multiply(previous_partial_gradient[:,j], self.input.data[:,i])
        self.weight.grad = np.sum(batch_w_grad, axis = 2)
        #self.weight.grad = np.mean(batch_w_grad, axis = 2)
        return np.matmul(previous_partial_gradient, self.weight.data.T)

    def selfstr(self):
        return str(self.weight.data.shape)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(LinearLayer, self).initialize()