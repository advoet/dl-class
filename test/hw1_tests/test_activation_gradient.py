import numpy as np
from nn.layers import *


def _test_backward_approx(layer, data_shape):
    h = 1e-4
    data = np.random.random(data_shape) * 10 - 5
    output1 = layer.forward(data + h)
    output2 = layer.forward(data - h)

    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output)

    output_gradient = layer.backward(previous_partial_gradient)

    assert np.allclose((output1 - output2) / (2 * h), output_gradient)


def test_layers():
    layers = [
        (LeakyReLULayer(0.001), (10, 20, 30)),
        (ReLULayer(), (10, 20, 30)),
        (PReLULayer(8, 0.001), (10, 8, 100)),
        (PReLULayer(1, 0.001), (10, 8, 100)),
    ]

    for layer, data_shape in layers:
        _test_backward_approx(layer, data_shape)
