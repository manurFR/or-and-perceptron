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

    def test_weights_stagnate_when_learning_rate_is_zero(self):
        perceptron = Perceptron(bias=0.5, weight_1=1.0, weight_2=1.0)
        perceptron.update_weights(learning_rate=0.0, input_1=0.4, input_2=0.6, actual_activation=1.0, expected_activation=0.0)
        self.assertEqual(0.5, perceptron.bias)
        self.assertEqual(1.0, perceptron.weight_1)
        self.assertEqual(1.0, perceptron.weight_2)

    def test_bias_is_updated_of_the_learning_rate_if_expected_is_different_than_actual(self):
        perceptron = Perceptron(bias=0.5, weight_1=1.0, weight_2=1.0)
        perceptron.update_weights(learning_rate=0.2, input_1=0.0, input_2=0.0, actual_activation=0.0, expected_activation=1.0)
        self.assertEqual(0.7, perceptron.bias)

    def test_inputs_are_updated_of_the_learning_rate_times_the_weight_if_expected_is_different_than_actual(self):
        perceptron = Perceptron(bias=0.5, weight_1=1.0, weight_2=1.0)
        perceptron.update_weights(learning_rate=0.3, input_1=1.0, input_2=1.0, actual_activation=1.0, expected_activation=0.0)
        self.assertEqual(0.7, perceptron.weight_1)
        self.assertEqual(0.7, perceptron.weight_2)

    def test_bias_and_inputs_are_not_changed_if_expected_is_equal_to_actual(self):
        perceptron = Perceptron(bias=0.5, weight_1=1.0, weight_2=0.8)
        perceptron.update_weights(learning_rate=0.3, input_1=1.0, input_2=1.0, actual_activation=1.0, expected_activation=1.0)
        self.assertEqual(0.5, perceptron.bias)
        self.assertEqual(1.0, perceptron.weight_1)
        self.assertEqual(0.8, perceptron.weight_2)



if __name__ == '__main__':
    unittest.main()
