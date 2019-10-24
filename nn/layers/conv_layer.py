from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        '''
        Each channel corresponds to a filter?. The first layer has 1 input channel (the original image) and we
        apply a # of filters = to # of output channels.

        Each filter consists of several kernels, which are what you think a filter is.

         - 32 x 32 raw image. 1 input channel, 3 (1x)1x1 filters (R, G, B)
        EXAMPLE:
         - 32 x 32 RGB image. First layer has 3 input channels (rgb), then height and width
         - A filter is then a stack of 3 kernels, one for each input channel, say 5x5
         - If we have 7 output channels, our weight matrix is 3 x 7 x 5 x 5
         - 3 x 5 x 5 is the filter (stack of 3 kernels)
         - 7 output channels, one for each filter.

        '''
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.W_row = np.zeros((output_channels, kernel_size*kernel_size*input_channels))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.X_col = None
        self.height = None
        self.width = None
        self.initialize()



    #@njit(parallel = True, cache = True)
    def weight_2_row(self, weights):
        '''
        Processes weight data into W_row format for the forward pass.

        There is one filter per output channel. We populate this matrix one row at a time
        '''
        def filter_to_row(filter_):
            # Filter is shape (# of kernels) x (kernel size) x (kernel size)
            whole_row = np.zeros(filter_.size)
            kernel_area = self.kernel_size * self.kernel_size
            for kernel in range(filter_.shape[0]):
                whole_row[kernel*kernel_area:(kernel+1)*kernel_area] = np.reshape(filter_[kernel,:,:],(-1)) 
            return whole_row

        for filter_ in range(0, weights.shape[1]):  
            whole_row = filter_to_row(weights[:,filter_,:,:])
            self.W_row[filter_, :] = whole_row


    #@njit(parallel = True, cache = True)
    def im_2_col(self, data):
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
            # gets the data from the square in data with top left corner (row, col)
            square = np.zeros((self.kernel_size, self.kernel_size))
            for i_square, i in enumerate(range(row, row+self.kernel_size)):
                for j_square, j in enumerate(range(col, col + self.kernel_size)):
                    if i < self.padding or i >= self.padding + data.shape[2] or j < self.padding or j >= self.padding + data.shape[3]:
                        square[i_square,j_square] = 0
                    else:
                        square[i_square,j_square] = data[n, channel, i-self.padding, j-self.padding]
            return square

        def to_column(square):
            return np.reshape(square, -1)

        def location_generator(height, width, kernel_size, padding, stride):
            ''' Generator for kernel placements on padded data '''
            x = 0
            y = 0
            while ((x + kernel_size - 1) < (height + 2*padding)):
                while((y + kernel_size - 1) < (width + 2*padding)):
                    yield (x,y)
                    y += stride
                y = 0
                x += stride

        for n in range(data.shape[0]):
            for channel in range(data.shape[1]):
                location = location_generator(self.height, self.width, self.kernel_size, self.padding, self.stride)
                for loc in range(self.X_col.shape[2]):
                    row, col = next(location)
                    self.X_col[n, channel*self.kernel_size*self.kernel_size:(channel+1)*self.kernel_size*self.kernel_size, loc] = to_column(get_square(row, col))
        # X_col has been populated

    def col_2_im(self, Y_col):
        '''
        The forward pass is calculated as W_row @ X_col = Y_col

        This method converts Y_col into the expected output format

        return: N (batch) x D (output channels) x H(eight) x W(idth)
        '''
        return Y_col.reshape((Y_col.shape[0],Y_col.shape[1], self.output_height, self.output_width))


    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias):
        # TODO
        # data is N (batch) x C (input channels) x H(eight) x W(idth)
        # kernel is COld x CNew x K (size) x K (size)
        for i in prange(data.shape[0]):
            self.im2col[i,:,:] = self.im_2_col(data, i)


    def forward(self, data):
        def number_of_locations(height, width, kernel_size, padding, stride):
            #too lazy for math
            horz_locations = 0
            loc = 0
            while((loc + kernel_size - 1) < (width + 2*padding)):
                horz_locations += 1
                loc += stride

            vert_locations = 0
            loc = 0
            while((loc + kernel_size - 1) < (height + 2*padding)):
                vert_locations +=1
                loc += stride
            self.output_width = horz_locations
            self.output_height = vert_locations
            return horz_locations*vert_locations

        self.height = data.shape[2]
        self.width = data.shape[3]

        size_size_channels = self.kernel_size*self.kernel_size*self.input_channels
        locations = number_of_locations(self.height, self.width, self.kernel_size, self.padding, self.stride)

        self.X_col = np.zeros((data.shape[0], size_size_channels, locations), dtype = np.float32)

        self.im_2_col(data)
        self.weight_2_row(self.weight.data)

        return self.col_2_im(self.W_row @ self.X_col)


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
