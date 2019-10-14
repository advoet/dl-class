import numpy as np
from numba import njit, prange

from .layer import Layer
from .linear_layer import LinearLayer


class ReLULayer(LinearLayer):
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))
        self.weight = Parameter(np.ones((input_size, output_size),dtype=np.float32))
        self.input = None
        self.forward_linear_data = None
        self.input_size = input_size
        self.output_size = output_size
        self.initialize()

    def forward(self, data):
        '''
        ReLU Layer forward pass

        :param data: n x d (batch x features)
        :return: n x c array (batch x channels) 
        '''
        self.forward_linear_data = Parameter(super().forward(data))
        return relu_activation(forward_linear_data.data)

    def backward(self, previous_partial_gradient):
        '''
        ReLU Layer backward pass

        :param previous_partial_gradient: n x c of gradients w.r.t. future layer
        :return: gradients w.r.t. inputs
        '''
        online_bias_grad = previous_partial_gradient * (self.forward_linear_data.data > 0)
        self.bias.grad = np.sum(online_bias_grad, axis = 0)

        batch_w_grad = np.zeros((self.input_size, self.output_size, np.size(previous_partial_gradient, 0)))
        for i in range(0, self.input_size):
            for j in range(0, self.output_size):
                batch_w_grad[i,j,:] = np.multiply(previous_partial_gradient[:,j], self.input.data[:,i]) * (self.forward_linear_data.data[:,j] > 0)
        self.weight.grad = np.sum(batch_w_grad, axis = 2)


        return np.matmul(previous_partial_gradient * (self.forward_linear_data.data > 0), self.weight.data.T)

    def relu_activation(forward_linear_data):
        relu_activated = forward_linear_data.copy()
        relu_activated *= forward_linear_data > 0
        return relu_activated

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
