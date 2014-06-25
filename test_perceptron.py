import unittest

from perceptron import Perceptron, initial_weights


class TestPerceptron(unittest.TestCase):
    def test_activate_on_perceptron_with_weights_equal_to_zero_returns_zero(self):
        perceptron = Perceptron(bias=0.0, weight_1=0.0, weight_2=0.0)
        activation = perceptron.activate(input_1=0.5, input_2=0.5)
        self.assertEqual(0.0, activation)

    def test_activate_with_inputs_equal_to_zero_returns_bias(self):
        perceptron = Perceptron(bias=0.5, weight_1=0.5, weight_2=0.5)
        activation = perceptron.activate(input_1=0.0, input_2=0.0)
        self.assertEqual(0.5, activation)

    def test_activate_works(self):
        perceptron = Perceptron(bias=0.5, weight_1=1.0, weight_2=0.8)
        activation = perceptron.activate(input_1=0.4, input_2=0.6)
        self.assertEqual(1.38, activation)

    def test_initial_weights(self):
        self.assertEqual({'bias': 0.1, 'weight_1': 0.1, 'weight_2': 0.1},
                         initial_weights(lambda: 0.1))

if __name__ == '__main__':
    unittest.main()
