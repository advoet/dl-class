from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .layer import Layer


class AddLayer(Layer):
    def __init__(self, parents):
        super(AddLayer, self).__init__(parents)
        self.input_channels = None

    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        # Need to count the total number of inputs for backward pass
        sum_out = np.zeros(1)
        for i, tensor in enumerate(inputs):
            sum_out += tensor
        self.input_channels = i
        return sum_out

    def backward(self, previous_partial_gradient): # -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        return tuple([previous_partial_gradient]*self.input_channels)
