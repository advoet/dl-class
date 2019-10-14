from .base_optimizer import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, regularization_constant = .9):
        super(SGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.regularization_constant = regularization_constant

    def step(self):
        for parameter in self.parameters:
            # TODO fix the line below to apply the parameter update
            # Now has some regularization
            parameter.data -= self.learning_rate*parameter.grad # + self.learning_rate*self.regularization_constant*parameter.data
