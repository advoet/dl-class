from .base_optimizer import BaseOptimizer


class MomentumSGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9, weight_decay=0):
        super(MomentumSGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.previous_deltas = [0] * len(parameters)

    def step(self):
        i = 0
        for parameter in self.parameters:
            # TODO update the parameters
            self.previous_deltas[i] = self.momentum * self.previous_deltas[i] + (parameter.grad + self.learning_rate * parameter.data)
            parameter.data -= self.learning_rate*self.previous_deltas[i]
            i += 1
