import random

WEIGHTS = ['bias', 'weight_1', 'weight_2']  # always in this order


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

    @property
    def weights(self):
        return [self.bias, self.weight_1, self.weight_2]


def initial_weights(random_func):
    return dict(zip(WEIGHTS, [random_func() for _ in range(3)]))


def convert(activation):
    return 1.0 if activation >= 0.5 else 0.0

# main
LEARNING_RATE = 0.1
OR_CASES = [([0.0, 0.0], 0.0),   # ([input_1, input_2], expected_value)
            ([1.0, 0.0], 1.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 1.0], 1.0)]

AND_CASES = [([0.0, 0.0], 0.0),   # ([input_1, input_2], expected_value)
             ([1.0, 0.0], 0.0),
             ([0.0, 1.0], 0.0),
             ([1.0, 1.0], 1.0)]

perceptron = Perceptron(**initial_weights(lambda: random.uniform(0.0, 0.5)))

iteration = 0
while True:
    iteration += 1
    print "* Iteration {}".format(iteration)
    weights_changed = False
    for inputs, expected in OR_CASES:  # replace by AND_CASES to find weights for AND
        actual = perceptron.activate(*inputs)
        print "weights = {} / inputs = {} / activation = {} ({}) / expected = {}".format(perceptron.weights, inputs, actual, convert(actual), expected)
        perceptron.update_weights(LEARNING_RATE, *inputs, actual_activation=convert(actual), expected_activation=expected)
        if convert(actual) != expected:
            weights_changed = True
            print " -> weights updated to {}".format(perceptron.weights)

    if not weights_changed:
        break

print
print "Converged ({} iterations)! Weights: {}".format(iteration, perceptron.weights)
