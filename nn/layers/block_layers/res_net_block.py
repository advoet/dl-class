from .. import *


class ResNetBlock(LayerUsingLayer):
    def __init__(self, conv_params, parent=None):
        super(ResNetBlock, self).__init__(parent)
        self.conv_layers = SequentialLayer([ConvLayer(*conv_params), ReLULayer(), ConvLayer(*conv_params)], parent)
        self.add_layer = None
        self.relu2 = None

    @property
    def final_layer(self):
        # TODO
        raise NotImplementedError

    def forward(self, data):
        # TODO
        return None
