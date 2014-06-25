import random


class Perceptron(object):

    def __init__(self, bias, weight_1, weight_2):
        self.bias = bias
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def activate(self, input_1, input_2):
        return self.bias + self.weight_1 * input_1 + self.weight_2 * input_2


def initial_weights(random_func):
    return dict(zip(['bias', 'weight_1', 'weight_2'], [random_func() for _ in range(3)]))

# main

perceptron = Perceptron(**initial_weights(lambda: random.uniform(0.0, 0.5)))