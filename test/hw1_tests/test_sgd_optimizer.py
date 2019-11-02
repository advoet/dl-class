import numpy as np
import torch

import nn
import nn.layers
from nn.optimizers import sgd_optimizer
from test import utils


class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.layer = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.layer(x)


def test_sgd_update():
    net = nn.layers.LinearLayer(100, 10)
    learning_rate = 1
    optimizer = sgd_optimizer.SGDOptimizer(net.parameters(), learning_rate)

    data = np.random.random((20, 100)).astype(np.float32) * 2 - 1

    initial_weight = net.weight.data.copy()
    initial_bias = net.bias.data.copy()

    torch_net = TorchNet()
    with torch.no_grad():
        torch_net.layer.weight[:] = utils.from_numpy(net.weight.data.T)
        torch_net.layer.bias[:] = utils.from_numpy(net.bias.data)
    torch_optimizer = torch.optim.SGD(torch_net.parameters(), learning_rate)

    optimizer.zero_grad()
    out = net(data)
    loss = out.sum()
    net.backward(np.ones_like(out))

    torch_optimizer.zero_grad()
    torch_out = torch_net(utils.from_numpy(data))
    assert np.allclose(out, utils.to_numpy(torch_out.clone().detach()), atol=0.001)

    torch_loss = torch_out.sum()
    assert np.allclose(loss, torch_loss.item(), atol=0.001)
    torch_loss.backward()

    assert np.allclose(net.weight.grad.T, utils.to_numpy(torch_net.layer.weight.grad))
    assert np.allclose(net.bias.grad, utils.to_numpy(torch_net.layer.bias.grad))

    optimizer.step()
    torch_optimizer.step()

    assert np.allclose(net.weight.data.T, utils.to_numpy(torch_net.layer.weight))
    assert np.allclose(net.bias.data, utils.to_numpy(torch_net.layer.bias))

    assert not np.allclose(net.weight.data, initial_weight)
    assert not np.allclose(net.bias.data, initial_bias)
