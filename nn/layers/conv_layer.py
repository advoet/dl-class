from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        '''
        Each channel corresponds to a filter. The first layer has 1 input channel (the original image) and we
        apply a # of filters = to # of output channels.

        Each filter consists of several kernels, which are what you think a filter is.

        '''
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.X_col = None
        self.initialize()

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias):
        # TODO
        # data is N (batch) x C (input channels) x H(eight) x W(idth)
        # kernel is COld x CNew x K (size) x K (size)
        for i in prange(data.shape[0]):
            self.im2col[i,:,:] = self.im_2_col(data, i)




    def forward(self, data):
        self.X_col = np.zeros((np.size(data,0)), self.kernel_size*self.kernel_size*self.input_channels, np.size(data,2)*np.size(data,3), dtype = np.float32)
        self.im_2_col(data)
        self.weight_2_row(self.weight)
        return self.W_row @ self.X_col 

    @njit(parallel = True, cache = True)
    def weight_2_row(self, data, index):
        self.W_row = 

        self.W_row = W_row

    @njit(parallel = True, cache = True)
    def im_2_col(self, data, index):
        '''
        Processes data into im2col form from lecture slides.
         - Data is not padded to begin with, so pad during conversion
         - Take into account stride as well
         - Each square analyzed by the filter becomes a column
         - In this manner, one channel (input image) is processed into a block row consisting of these columns

        final size is:
         N (batch) x size*size*channels x total number of pixels
        '''
        def get_square(row, col):
            square = np.zeros((self.kernel_size, self.kernel_size))
            for i in range(row, row+self.kernel_size):
                for j in range(col, col + self.kernel_size):
                    if i < self.padding 
                            or i > self.padding + data.shape[3]
                            or j < self.padding
                            or j > self.padding + data.shape[4]:
                        square[i,j] = 0
                    else:
                        square[i,j] = data[n, channel], i, j]
            return square

        def to_column(square):
            return np.reshape(a, (-1, 1))

        for n in prange(0, data.shape[0]) # batch size
            for channel in prange(0, data.shape[1])
                for row in prange(0, data.shape[3] + 2*self.padding - self.kernel_size) # across columns with padding:
                    for col in prange(0, data.shape[4] + 2*self.padding - self.kernel_size) 
                        self.X_col[n, row, col] = to_column(get_square(row, col))
        # X_col has been populated


        
        
    #DO WE NEED A COL_2_IM AS WELL? PROBABLY

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
