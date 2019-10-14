from typing import Optional, Callable

import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):

    iters = 0

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
        #-
        #LinearLayer.iters+=1
        #print(LinearLayer.iters)
        #print(self.weight.grad)
        #
        #ha! my weights were exploding so I used regularization, but it didnt work T_T
        #
        #self.input = Parameter(data.astype(np.float64))
        #
        return np.matmul(self.input.data, self.weight.data) + self.bias.data

    def backward(self, previous_partial_gradient: np.ndarray): #-> np.ndarray
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future layer
        :return: gradients wrt inputs
        """
        
        self.bias.grad = np.sum(previous_partial_gradient, axis = 0)
        self.weight.grad = np.matmul(self.input.data.T, previous_partial_gradient)
        
        return np.matmul(previous_partial_gradient, self.weight.data.T, dtype = np.double)

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