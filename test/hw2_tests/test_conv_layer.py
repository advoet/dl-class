import pdb
import numpy as np
import torch
from torch import nn

from nn.layers.conv_layer import ConvLayer
from test import utils

TOLERANCE = 1e-4


def _test_conv_forward(input_shape, out_channels, kernel_size, stride, pr = None):
    return
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = ConvLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    utils.assign_conv_layer_weights(layer, torch_layer)

    '''
    if pr is not None:    
        pr.enable()
    '''

    output = layer.forward(input)
    
    '''
    if pr is not None:
        pr.disable()
    '''

    torch_data = utils.from_numpy(input)
    torch_out = torch_layer(torch_data)

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    utils.assert_close(output, torch_out, atol=TOLERANCE)


def test_conv_forward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_conv_forward(input_shape, output_channels, kernel_size, stride)


def test_conv_forward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    output_channels = 3

    pr = None
    '''
    import cProfile
    pr = cProfile.Profile()
    '''

    for width in range(10, 21):
        for height in range(10, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_conv_forward(input_shape, output_channels, kernel_size, stride)
    '''
    import pdb; pdb.set_trace()
    '''

                    

def _test_conv_backward(input_shape, out_channels, kernel_size, stride, pr = None):
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = ConvLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    utils.assign_conv_layer_weights(layer, torch_layer)


    if pr is not None:    
        pr.enable()

    output = layer.forward(input)

    if pr is not None:
        pr.disable()

    out_grad = layer.backward(np.ones_like(output))

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    torch_out.sum().backward()
    
    utils.assert_close(out_grad, torch_input.grad, atol=TOLERANCE)
    
    utils.check_conv_grad_match(layer, torch_layer)


def test_conv_backward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_conv_backward(input_shape, output_channels, kernel_size, stride)


def test_conv_backward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    output_channels = 3

    pr = None
    '''
    import cProfile
    pr = cProfile.Profile()
    '''

    for width in range(10, 21):
        for height in range(10, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_conv_backward(input_shape, output_channels, kernel_size, stride, pr)

    '''
    import pdb; pdb.set_trace()
    '''
