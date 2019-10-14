import numpy as np
import torch
from torch import nn

from nn.layers.linear_layer import LinearLayer
from test import utils

TOLERANCE = 1e-4


def _test_linear_forward(input_shape, out_channels):
    in_channels = input_shape[1]
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = LinearLayer(in_channels, out_channels)

    torch_layer = nn.Linear(in_channels, out_channels, bias=True)
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)

    output = layer.forward(input)

    torch_data = utils.from_numpy(input)
    torch_out = utils.to_numpy(torch_layer(torch_data))
    output[np.abs(output) < 1e-4] = 0
    torch_out[np.abs(torch_out) < 1e-4] = 0

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    assert np.allclose(output, torch_out, atol=TOLERANCE)


def test_linear_forward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels)
                _test_linear_forward(input_shape, output_channels)


def _test_linear_backward(input_shape, out_channels,):

    in_channels = input_shape[1]
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = LinearLayer(in_channels, out_channels)

    torch_layer = nn.Linear(in_channels, out_channels, bias=True)
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)

    output = layer.forward(input)
    out_grad = layer.backward(np.ones_like(output))

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    torch_out.sum().backward()

    torch_out_grad = utils.to_numpy(torch_input.grad)
    out_grad[np.abs(out_grad) < 1e-4] = 0
    torch_out_grad[np.abs(torch_out_grad) < 1e-4] = 0
    assert np.allclose(out_grad, torch_out_grad, atol=TOLERANCE)

    w_grad = layer.weight.grad
    w_grad[np.abs(w_grad) < 1e-4] = 0
    torch_w_grad = utils.to_numpy(torch_layer.weight.grad.transpose(0, 1))
    torch_w_grad[np.abs(torch_w_grad) < 1e-4] = 0

    print(w_grad)
    print(torch_w_grad)
    print()
    print("--------------")
    print()

    assert np.allclose(w_grad, torch_w_grad, atol=TOLERANCE)

    b_grad = layer.bias.grad
    b_grad[np.abs(b_grad) < 1e-4] = 0
    torch_b_grad = utils.to_numpy(torch_layer.bias.grad)
    torch_b_grad[np.abs(torch_b_grad) < 1e-4] = 0
    assert np.allclose(b_grad, torch_b_grad, atol=TOLERANCE)


def test_linear_backward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels)
                _test_linear_backward(input_shape, output_channels)
