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

    def row_2_weight(self, W_row):
        unrow = W_row.reshape((W_row.shape[0],self.input_channels,self.kernel_size,self.kernel_size))
        return np.swapaxes(unrow, 0, 1)

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
            # deals with the padding. Probably faster to just use numpy pad
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


        for n in range(data.shape[0]):
            for channel in range(data.shape[1]):
                locations = ConvLayer.location_generator(self.height, self.width, self.kernel_size, self.padding, self.stride)
                for index, (row, col) in enumerate(locations):
                    self.X_col[n, channel*self.kernel_size*self.kernel_size:(channel+1)*self.kernel_size*self.kernel_size, index] = to_column(get_square(row, col))

    @staticmethod
    def location_generator(height, width, kernel_size, padding, stride):
            ''' Generator for kernel placements on padded data

            :return: (x,y) row column location of top left corner of kernel
            '''
            x = 0
            y = 0
            while ((x + kernel_size - 1) < (height + 2*padding)):
                while((y + kernel_size - 1) < (width + 2*padding)):
                    yield (x,y)
                    y += stride
                y = 0
                x += stride

    def col_2_im(self, data_col, height, width):
        '''
        Used in backward pass

        :param data_col: numpy array (N, size_size_channels, locations)
        :param height, width: dimensions of true image (no padding)


        '''
        batch_size = data_col.shape[0]
        input_channels = self.input_channels
        size = self.kernel_size

        def full_batch_all_channels_square_from(location_index):
            return data_col[:, :, location_index].reshape((batch_size, input_channels, size, size))

        padded_image_grad = np.zeros((data_col.shape[0], self.input_channels, self.height+2*self.padding, self.width+2*self.padding))
        
        kernel_locations = ConvLayer.location_generator(self.height, self.width, self.kernel_size, self.padding, self.stride)
        for index, (row, col) in enumerate(kernel_locations):
            padded_image_grad[
                              :,
                              :,
                              row:row+self.kernel_size,
                              col:col+self.kernel_size,
                              ] = full_batch_all_channels_square_from(index)
        
        return padded_image_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def number_of_locations(self, height, width, kernel_size, padding, stride):
        horz_loc = 1 + ((width + 2*padding - kernel_size)//stride)
        vert_loc = 1 + ((height + 2*padding - kernel_size)//stride)
        self.output_width = horz_loc
        self.output_height = vert_loc
        return horz_loc*vert_loc

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias):
        # TODO
        # data is N (batch) x C (input channels) x H(eight) x W(idth)
        # kernel is COld x CNew x K (size) x K (size)
        for i in prange(data.shape[0]):
            self.im2col[i,:,:] = self.im_2_col(data, i)

    def forward(self, data):

        self.height = data.shape[2]
        self.width = data.shape[3]

        size_size_channels = self.kernel_size*self.kernel_size*self.input_channels
        self.location_count = self.number_of_locations(self.height, self.width, self.kernel_size, self.padding, self.stride)

        self.X_col = np.zeros((data.shape[0], size_size_channels, self.location_count), dtype = np.float32)
        self.im_2_col(data)
        
        W_row = np.reshape(np.swapaxes(self.weight.data,0,1), (self.weight.data.shape[1],-1))
        Y_col = W_row @ self.X_col
        return Y_col.reshape((*Y_col.shape[0:2], self.output_height, self.output_width)) + self.bias.data.reshape((1,self.output_channels,1,1))


    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, X_col, weight, weight_grad):
        # TODO
        # data is already in column form
        # kernel is COld x CNew x K x K
        pass

    def backward(self, previous_partial_gradient):
        '''
        :param previous_partial_gradient: N x D x H x W

        '''
        self.bias.grad = np.sum(previous_partial_gradient, axis = (0,2,3))

        delta = previous_partial_gradient.reshape(*previous_partial_gradient.shape[0:2], -1)
        W_row_batch_grad = sum(delta[i,:,:] @ self.X_col[i,:,:].T for i in range(self.X_col.shape[0]))
        self.weight.grad = self.row_2_weight(W_row_batch_grad)

        ########
        # Everything above this line works
        ########
        W_row = np.reshape(np.swapaxes(self.weight.data,0,1), (self.weight.data.shape[1],-1))
        previous_grad_col = previous_partial_gradient.reshape((*previous_partial_gradient.shape[0:2],-1))

        next_grad_col = W_row.T @ previous_grad_col

        return self.col_2_im(next_grad_col, self.height, self.width)

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
