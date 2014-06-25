import random


class Perceptron(object):
    def __init__(self, bias, weight_1, weight_2):
        self.bias = bias
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def activate(self, input_1, input_2):
        return self.bias + self.weight_1 * input_1 + self.weight_2 * input_2

    def update_weights(self, learning_rate, input_1, input_2, actual_activation, expected_activation):
        evaluation = expected_activation - actual_activation
        self.bias     += learning_rate * evaluation * 1.0
        self.weight_1 += learning_rate * evaluation * input_1
        self.weight_2 += learning_rate * evaluation * input_2


def initial_weights(random_func):
    return dict(zip(['bias', 'weight_1', 'weight_2'], [random_func() for _ in range(3)]))

# main

perceptron = Perceptron(**initial_weights(lambda: random.uniform(0.0, 0.5)))