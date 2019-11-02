import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import tqdm
import time
import numpy as np

import cProfile

from nn import layers
from nn.layers import losses
from nn.optimizers.momentum_sgd_optimizer import MomentumSGDOptimizer
from nn import Network
from nn.layers.block_layers import ResNetBlock


class MNISTNetwork(Network):
    def __init__(self):
        self.layers = layers.SequentialLayer(
            [
                layers.ConvLayer(1, 6, 5),
                layers.MaxPoolLayer(2, 2),
                layers.LeakyReLULayer(),
                layers.ConvLayer(6, 16, 5),
                layers.MaxPoolLayer(2, 2),
                layers.LeakyReLULayer(),
                layers.FlattenLayer(),
                layers.LinearLayer(16 * 7 * 7, 120),
                layers.LeakyReLULayer(),
                layers.LinearLayer(120, 84),
                layers.LeakyReLULayer(),
                layers.LinearLayer(84, 10),
            ]
        )
        loss_layer = losses.SoftmaxCrossEntropyLossLayer(parent=self.layers)
        super(MNISTNetwork, self).__init__(loss_layer)

    def forward(self, data):
        return self.layers(data)

    def loss(self, predictions, labels):
        return self.loss_layer(predictions, labels)


class MNISTResNetwork(Network):
    def __init__(self):
        self.layers = layers.SequentialLayer(
            [
                layers.ConvLayer(1, 6, 5),
                layers.MaxPoolLayer(2, 2),
                layers.LeakyReLULayer(),
                layers.ConvLayer(6, 16, 5),
                ResNetBlock((16, 16, 3, 1)),
                ResNetBlock((16, 16, 3, 1)),
                layers.MaxPoolLayer(2, 2),
                layers.LeakyReLULayer(),
                layers.FlattenLayer(),
                layers.LinearLayer(16 * 7 * 7, 120),
                layers.LeakyReLULayer(),
                layers.LinearLayer(120, 84),
                layers.LeakyReLULayer(),
                layers.LinearLayer(84, 10),
            ]
        )
        loss_layer = losses.SoftmaxCrossEntropyLossLayer(parent=self.layers)
        super(MNISTResNetwork, self).__init__(loss_layer)

    def forward(self, data):
        return self.layers(data)

    def loss(self, predictions, labels):
        return self.loss_layer(predictions, labels)


def train(train_data, train_labels, test_data, test_labels, network):
    optimizer = MomentumSGDOptimizer(network.parameters(), lr, weight_decay=.0005)
    print(network)


    pr = cProfile.Profile()


    iteration = -1
    print("-" * 50)
    
    iteration = -1
    epoch = 0
    for epoch in range(20):
        for ii in tqdm.tqdm(range(0, len(train_data), batch_size)):
            iteration += 1
            data = train_data[ii : min(ii + batch_size, len(train_data))]
            labels = train_labels[ii : min(ii + batch_size, len(train_data))]
            optimizer.zero_grad()
            
            pr.enable()
            output = network(data)
            pr.disable()

            accuracy = (np.argmax(output, 1) == labels).mean()
            loss = network.loss(output, labels)
            if iteration % 25 == 0:
                print("step", iteration, "train accuracy %.3f" % accuracy, "loss %.3f" % loss)
                import pdb; pdb.set_trace()
                # pr.print_stats(sort = 'tottime')
            t_start = time.time()

            pr.enable()
            network.backward()
            pr.disable()
            # print('backward end %.3f' % (time.time() - t_start))
            optimizer.step()
        epoch += 1
        output = network(test_data)
        accuracy = (np.argmax(output, 1) == test_labels).mean()
        loss = network.loss(output, test_labels)
        print("-" * 50)
        print("\tepoch", epoch, "test accuracy %.3f" % accuracy, "loss %.3f" % loss)
    print("done")

if __name__ == "__main__":
    batch_size = 100
    lr = 0.01

    train_dataset = np.load("../datasets/mnist/train.npz")
    train_data = train_dataset["data"].astype(np.float32) / 255
    train_data = train_data[:, np.newaxis, ...]
    train_labels = train_dataset["labels"]

    test_dataset = np.load("../datasets/mnist/test.npz")
    test_data = test_dataset["data"].astype(np.float32) / 255
    test_data = test_data[:, np.newaxis, ...]
    test_labels = test_dataset["labels"]

    network = MNISTResNetwork()
    train(train_data, train_labels, test_data, test_labels, network)
