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
        self.batch_size = None
        self.input_channels = None
        self.height = None
        self.width = None
        self.output_height = None
        self.output_width = None
        self.location_count = None
        self.locations = None
        self.NC_argmaxes = None
        self.NC_X_col = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # data is N x C x H x W
        # TODO
        return None

    def forward(self, data):

        self.batch_size = data.shape[0]
        self.input_channels = data.shape[1]
        self.height = data.shape[2]
        self.width = data.shape[3]

        #DON'T DELETE: also initializes output width and height
        self.location_count = self.number_of_locations(self.height,
                                                        self.width,
                                                        self.kernel_size,
                                                        self.padding,
                                                        self.stride)

        self.locations = [loc for loc in MaxPoolLayer.location_generator(self.height,
                                                                self.width,
                                                                self.kernel_size,
                                                                self.padding,
                                                                self.stride)]
            
        NC_data = data.reshape(self.batch_size*self.input_channels,1,self.height,self.width)
        
        self.NC_X_col = self.im_2_col_pad(NC_data)
        self.NC_argmaxes = np.argmax(self.NC_X_col, axis = 1)
        NC_maxes = np.max(self.NC_X_col, axis = 1)
        maxpool = NC_maxes.reshape((self.batch_size, self.input_channels, self.output_height, self.output_width))

        return maxpool

    #############
    # METHODS FOR FORWARD PASS
    #############

    def number_of_locations(self, height, width, kernel_size, padding, stride):
        horz_loc = 1 + ((width + 2*padding - kernel_size)//stride)
        vert_loc = 1 + ((height + 2*padding - kernel_size)//stride)
        self.output_width = horz_loc
        self.output_height = vert_loc
        return horz_loc*vert_loc

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

    def im_2_col_pad(self, data):
        '''
        See im_2_col. This method pads the data first instead of relegating to get square
        '''
        size = self.kernel_size
        padding = self.padding
        data = np.pad(data,((0,0),(0,0),(padding,padding),(padding,padding)))

        long_X_col = np.zeros((data.shape[0], size*size, self.location_count), dtype = np.float32)
        for n in range(data.shape[0]):
            for channel in range(data.shape[1]):
                for index, (row, col) in enumerate(self.locations):
                    #slowwwww HOW TO DO THIS WITHOUT RESHAPING EVERY TIME
                    long_X_col[
                               n,
                               channel*size*size:(channel+1)*size*size,
                               index
                               ] \
                               = data[
                                      n,
                                      channel,
                                      row:row + size,
                                      col:col + size
                                     ].reshape(size*size)
        return long_X_col

    #############
    # END OF METHODS FOR FORWARD PASS
    #############
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data):
        # data is N x C x H x W
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        '''
        :arg previous_partial_gradient: N x C x output_height x output_width

        dL/dx_ij += 1 if x_ij is the max in a box, 0 otherwise

        self.NC_X_col[NC_row,self.NC_argmaxes[i,loc],loc] is the max entry.





        NC_row == 1 corresponds to batch 0                  channel 1
        NC_row == i corresponds to batch i//input_channels  channel i'%'input_channels

        don't care about the entry, just the coordinate in the original shebang

        '''
        def _position_finder(batch_channel, loc, p):
            # IMPROVEMENTS - 
            # returns the row and column of NC_X_col[batch_channel, loc]
            # in the original matrix
            row, col = self.locations[loc]
            #surely can use fancy indexing
            column_index = self.NC_argmaxes[batch_channel, loc]
            r, c = divmod(column_index, self.kernel_size)
            return row + r - p, col + c - p

        next_gradient = np.zeros((self.batch_size,
                                  self.input_channels,
                                  self.height,
                                  self.width))

        #looping over NC_X_argmaxes
        p = self.padding
        k = self.kernel_size
        for batch_channel in range(self.NC_argmaxes.shape[0]):
            for loc in range(self.NC_argmaxes.shape[1]):
                batch, channel = divmod(batch_channel, self.input_channels)
                row, col = _position_finder(batch_channel, loc, p)
                #image is square
                out_row, out_col = divmod(loc, self.output_height)

                #need to handle out of bounds indices
                try:
                    next_gradient[batch,
                                    channel,
                                    row,
                                    col] += previous_partial_gradient[batch,
                                                                        channel,
                                                                        out_row,
                                                                        out_col]
                except IndexError:
                    # ignore the padding
                    pass

        return next_gradient

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
