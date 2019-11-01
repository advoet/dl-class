from .. import *


class ResNetBlock(LayerUsingLayer):
    def __init__(self, conv_params, parent=None):
        super(ResNetBlock, self).__init__(parent)
        self.conv_layers = SequentialLayer([ConvLayer(*conv_params),
                                            ReLULayer(),
                                            ConvLayer(*conv_params)],
                                                self.parent)
        self.add_layer = AddLayer((self.conv_layers.final_layer, self.conv_layers.parent))
        self.relu2 = ReLULayer(self.add_layer)

    @property
    def final_layer(self):
        return self.relu2

    def forward(self, data):
        conv_data = self.conv_layers.forward(data)
        return self.relu2.forward(self.add_layer.forward((data, conv_data)))