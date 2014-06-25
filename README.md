OR/AND Perceptron
=================

Python implementation of the [Perceptron](http://en.wikipedia.org/wiki/Perceptron), as discovered by Frank Rosenblatt in 1957.

A Perceptron is a simplified simulation of a single neural cell (a neuron) that takes a series of input signals and outputs a corresponding value. Such neurons are the building blocks of (artificial) neural networks.

As [described by Jason Brownlee](http://www.cleveralgorithms.com/nature-inspired/neural/perceptron.html),
"The information processing objective of the technique is to model a given function by modifying internal weightings of input signals to produce an expected output signal. The system is trained using a supervised learning method, where the error between the system's output and a known expected output is presented to the system and used to modify its internal state. State is maintained in a set of weightings on the input signals. The weights are used to represent an abstraction of the mapping of input vectors to the output signal for the examples that the system was exposed to during training."

This is a Perceptron that we will train to reach working weights in order to simulate OR and AND gates. The decision matrix has 2 inputs in both cases. Thus this is probably the simplest Perceptron in the world (excluding a useless "identity Perceptron").
