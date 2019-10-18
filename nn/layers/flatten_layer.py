from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.input_channels = None
        self.height = None
        self.width = None

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        self.input_channels, self.height, self.width = data.shape[1], data.shape[2], data.shape[3]
        return np.reshape(data, (data.shape[0],-1))

    def backward(self, previous_partial_gradient):
        # TODO
        return np.reshape(previous_partial_gradient, (previous_partial_gradient.shape[0], self.input_channels, self.height, self.width))
