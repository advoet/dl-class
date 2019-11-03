from typing import Optional, Callable
import numpy as np

from numba import jit,njit, prange

from itertools import product

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


    ###############
    #### Various im_2_col implementations
    ###############

    # im_2_col -- works, uses get_square instead of padding
    # im_2_col_pad  -- TO DO, use np.pad, maybe a faster get square
    # im_2_col_no_sq -- [Zhang '17 Simple and Efficient Implementation]
    # im_2_col_numba -- 



    @staticmethod
    @njit(parallel=True, cache=True)
    def numba_im_2_col(X, H0, H1, W0, W1, D0, D1, P, K, S):
        '''
        trying to parallelize
        See im_2_col_no_sq
        '''
        X_col = np.zeros((X.shape[0], K*K*D0, H1*W1), dtype=np.float32)
        

        for l in prange(D0*K*K*H1*W1):
            # p gives the channel
            p = l // (H1*W1)
            # q gives the location
            q = l % (H1*W1)
            # d0 is the row in X_col
            d0 = (p//K)//K
            i0 = S*(q//W1) + (p//K) %K
            j0 = S*(q % W1) + p % K
            for n in prange(X.shape[0]):
                if (i0 >= P and j0 >= P and i0 < H0 + P and j0 < W0 + P):
                    X_col[n,p,q] = X[n,d0,i0-P,j0-P]
                else:
                    X_col[n,p,q] = 0
        return X_col

    
    @staticmethod
    @njit(parallel=True, cache=True)
    def numba_im_2_col_2(X, H0, H1, W0, W1, D0, P, K, S):
        '''
        trying to parallelize
        See im_2_col_no_sq
        '''
        X_col = np.zeros((X.shape[0], K*K*D0, H1*W1), dtype=np.float32)

        for loc in prange(H1*W1):
            tl_r = S*(loc // W1)
            tl_c = S*(loc % W1)
            for row in prange(K*K*D0):
                d0 = row//(K*K)
                r0 = row % (K*K)
                i0 = tl_r + (r0 // K)
                j0 = tl_c + (r0 % K)
                
                for n in prange(X.shape[0]):
                    if (i0 >= P and j0 >= P and i0 < H0 + P and j0 < W0 + P):
                        X_col[n,row,loc] = X[n,d0,i0-P,j0-P]
                    else:
                        X_col[n,row,loc] = 0
        return X_col

    @staticmethod
    @njit(parallel=True, cache=True)
    def numba_col_2_im(X_col, H0, H1, W0, W1, K, P, S, D0):
        # cannot parallelize all loops bc of racing
        # njit does not like 0 padding for some reason

        X_im = np.zeros((X_col.shape[0], D0, H0+2*P, W0+2*P))        
        for loc in range(H1*W1):
            tl_r = S*(loc // W1)
            tl_c = S*(loc % W1)
            for chan in prange(D0):
                for row in range(K*K*chan, K*K*(chan+1)):
                    r0 = row % (K*K)
                    i0 = tl_r + (r0 // K)
                    j0 = tl_c + (r0 % K)
                    for n in prange(X_im.shape[0]):
                        X_im[n,chan,i0,j0] += X_col[n,row,loc]
        return X_im

    def im_2_col_indexing(self, data):
        size = self.kernel_size
        C = self.input_channels
        padding = self.padding
        data = np.pad(data,((0,0),(0,0),(padding,padding),(padding,padding)))

        locations = [loc for loc in ConvLayer.location_generator(self.height,
                                                                 self.width,
                                                                 size,
                                                                 padding,
                                                                 self.stride)]

        def channel_square_indices(row, col):
            # in a given column of X_col, corresponding to a location
            # return a column of indices in padded_data (c, h,k)
            # channel_square_indices(row,col)[3] is entry in third row at loc
            k = self.kernel_size
            c = self.input_channels
            return [(channel, x, y) for channel in range(c)
                                    for (x,y) in product(range(row, row+k),
                                                         range(col,col+k))]
        index_map = []                              
        for loc, (row,col) in enumerate(locations):
            index_map.append([])
            for channel_square_index in channel_square_indices(row, col):
                index_map[loc].append(channel_square_index)
        

        X_col[:, r, loc] = padded_data[:, ]



    
    def im_2_col_no_sq(self, data):
        '''
        See im_2_col. This method from [Zhang '17]

        Implementation and stride count.
        Next step is split up the k loop for parallel computation
        '''
        H0 = self.height
        H1 = self.output_height
        W0 = self.width
        W1 = self.output_width
        D0 = self.input_channels
        D1 = self.output_channels
        P = self.padding
        K = self.kernel_size
        S = self.stride
        X = data
        for k in range(D0*K*K*H1*W1):
            p = k // (H1*W1)
            q = k % (H1*W1)
            d0 = (p//K)//K
            i0 = S*(q//W1) + (p//K) %K
            j0 = S*(q % W1) + p % K
            if (i0 >= P and j0 >= P and i0 < H0 + P and j0 < W0 + P):
                self.X_col[:,p,q] = X[:,d0,i0-P,j0-P]
            else:
                self.X_col[:,p,q] = 0


        

    def im_2_col_pad(self, data):
        '''
        See im_2_col. This method pads the data first instead of relegating to get square
        '''
        size = self.kernel_size
        padding = self.padding
        data = np.pad(data,((0,0),(0,0),(padding,padding),(padding,padding)))

        # saves a whole second to do this outside instead of using a generator
        # I guess locations don't take that much space.
        locations = [loc for loc in ConvLayer.location_generator(self.height,
                                                                 self.width,
                                                                 size,
                                                                 padding,
                                                                 self.stride)]
        for n in range(data.shape[0]):
            for channel in range(data.shape[1]):
                for index, (row, col) in enumerate(locations):
                    #HOW TO DO THIS WITHOUT RESHAPING EVERY TIME
                    self.X_col[
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

    @staticmethod
    def location_generator2(height, width, kernel_size, padding, stride):
            ''' Generator for kernel placements on padded data

            :return: (x,y) row column location of top left corner of kernel
            '''
            x = 0
            y = 0
            while ((x + kernel_size - 1) < (height + 2*padding)):
                while((y + kernel_size - 1) < (width + 2*padding)):
                    yield [x,y]
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
        
        kernel_locations = [loc for loc in ConvLayer.location_generator(self.height,
                                                                        self.width,
                                                                        self.kernel_size,
                                                                        self.padding,
                                                                        self.stride)]
        for index, (row, col) in enumerate(kernel_locations):
            padded_image_grad[
                              :,
                              :,
                              row:row+self.kernel_size,
                              col:col+self.kernel_size,
                              ] += full_batch_all_channels_square_from(index)
        
        
        if self.padding == 0:
            return padded_image_grad
        else:
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

        #also initializes output height and width
        self.location_count = self.number_of_locations(self.height, self.width, self.kernel_size, self.padding, self.stride)

        self.X_col = np.zeros((data.shape[0], size_size_channels, self.location_count), dtype = np.float32)
        
        ###########################
        # IM2COL GOES HERE
        ###########################

        #Original line:
        #self.im_2_col(data)

        #self.im_2_col_no_sq(data)
      
        # With numpy padding:
        # self.im_2_col_pad(data)
        

        # other fance
        #self.im_2_col_indexing(data)

        
        #numba code
        self.X_col = ConvLayer.numba_im_2_col_2(data,
                                              self.height,
                                              self.output_height,
                                              self.width,
                                              self.output_width,
                                              self.input_channels,
                                              self.padding,
                                              self.kernel_size,
                                              self.stride)
        
        

        ###########################

        W_row = np.reshape(np.swapaxes(self.weight.data,0,1), (self.weight.data.shape[1],size_size_channels))
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

        W_row = np.reshape(np.swapaxes(self.weight.data,0,1), (self.weight.data.shape[1],-1))
        previous_grad_col = previous_partial_gradient.reshape((*previous_partial_gradient.shape[0:2],-1))

        next_grad_col = W_row.T @ previous_grad_col

        #return self.col_2_im(next_grad_col, self.height, self.width)
        X_im_pad = ConvLayer.numba_col_2_im(next_grad_col,
                                              self.height,
                                              self.output_height,
                                              self.width,
                                              self.output_width,
                                              self.kernel_size,
                                              self.padding,
                                              self.stride, 
                                              self.input_channels)
        p = self.padding
        if p is 0:
            return X_im_pad
        else:
            return X_im_pad[:,:,p:-p, p:-p]

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
