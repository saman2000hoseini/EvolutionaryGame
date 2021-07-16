import math

import numpy as np


def relu(z):
    if z < 0:
        return 0
    return z


def sigmoid(z):
    if z < 0:
        return 1 - 1 / (1 + math.exp(z))
    return 1 / (1 + math.exp(-z))


def activation(z):
    return sigmoid(z)


def feed_forward(w, a, b):
    z = np.matmul(w, a) + b
    return activation(z), z


activation = np.vectorize(activation)


class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.b0 = np.random.randn(layer_sizes[1], 1)
        self.b1 = np.random.randn(layer_sizes[1], 1)
        self.b2 = np.random.randn(layer_sizes[2], 1)

        self.w0 = np.random.randn(layer_sizes[1], layer_sizes[0])
        self.w1 = np.random.randn(layer_sizes[1], layer_sizes[1])
        self.w2 = np.random.randn(layer_sizes[2], layer_sizes[1])

    def forward(self, x):
        a0 = np.array(x).reshape((len(x), 1))
        a1, z1 = feed_forward(self.w0, a0, self.b0)
        a2, z2 = feed_forward(self.w1, a1, self.b1)
        a3, z3 = feed_forward(self.w2, a2, self.b2)

        return a3
