from collections import defaultdict

import numpy as np
import torch

TOLERANCE = 1e-4


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, dict):
        return {key: to_numpy(val) for key, val in array.items()}
    else:
        return np.asarray(array)


numpy_dtype_to_pytorch_dtype_warn = False


def numpy_dtype_to_pytorch_dtype(numpy_dtype):
    global numpy_dtype_to_pytorch_dtype_warn
    # Extremely gross conversion but the only one I've found
    numpy_dtype = np.dtype(numpy_dtype)
    if numpy_dtype == np.uint32:
        if not numpy_dtype_to_pytorch_dtype_warn:
            print("numpy -> torch dtype uint32 not supported, using int32")
            numpy_dtype_to_pytorch_dtype_warn = True
        numpy_dtype = np.int32
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).detach().dtype


from_numpy_warn = defaultdict(lambda: False)


def from_numpy(np_array):
    global from_numpy_warn
    if isinstance(np_array, list):
        try:
            np_array = np.stack(np_array, 0)
        except ValueError:
            np_array = np.stack([from_numpy(val) for val in np_array], 0)
    elif isinstance(np_array, dict):
        return {key: from_numpy(val) for key, val in np_array.items()}
    np_array = np.asarray(np_array)
    if np_array.dtype == np.uint32:
        if not from_numpy_warn[np.uint32]:
            print("numpy -> torch dtype uint32 not supported, using int32")
            from_numpy_warn[np.uint32] = True
        np_array = np_array.astype(np.int32)
    elif np_array.dtype == np.dtype("O"):
        if not from_numpy_warn[np.dtype("O")]:
            print("numpy -> torch dtype Object not supported, returning numpy array")
            from_numpy_warn[np.dtype("O")] = True
        return np_array
    elif np_array.dtype.type == np.str_:
        if not from_numpy_warn[np.str_]:
            print("numpy -> torch dtype numpy.str_ not supported, returning numpy array")
            from_numpy_warn[np.str_] = True
        return np_array
    return torch.from_numpy(np_array)


def assign_linear_layer_weights(layer, torch_layer):
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)


def assign_conv_layer_weights(layer, torch_layer):
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)


def allclose(val1, val2, atol=TOLERANCE):
    val1 = to_numpy(val1)
    val2 = to_numpy(val2)
    val1[np.abs(val1) < 1e-4] = 0
    val2[np.abs(val2) < 1e-4] = 0
    return np.allclose(val1, val2, atol=atol)


def check_linear_grad_match(layer, torch_layer, tolerance=TOLERANCE):
    w_grad = layer.weight.grad
    torch_w_grad = to_numpy(torch_layer.weight.grad.transpose(0, 1))
    assert allclose(w_grad, torch_w_grad, tolerance)
    b_grad = layer.bias.grad
    torch_b_grad = to_numpy(torch_layer.bias.grad)
    assert allclose(b_grad, torch_b_grad, tolerance)


def check_conv_grad_match(layer, torch_layer, tolerance=TOLERANCE):
    w_grad = layer.weight.grad
    torch_w_grad = to_numpy(torch_layer.weight.grad.transpose(0, 1))
    assert allclose(w_grad, torch_w_grad, tolerance)
    b_grad = layer.bias.grad
    torch_b_grad = to_numpy(torch_layer.bias.grad)
    assert allclose(b_grad, torch_b_grad, tolerance)
