import numpy as np


def sigmoid(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):  # sigmoid'
    # sigmoid'(s) = sigmoid(s) * (1 - sigmoid(s)) = y * (1 - y)
    return y * (1 - y)


def linear(s):  # linear
    return s


def d_linear(y):  # linear'
    return 1.0


def relu_0(s):
    return s if s > 0 else 0


def d_relu_0(y):
    return 1 if y > 0 else 0


class Relu:
    def __init__(self, k):
        """ReLu с параметром k"""
        self.k = k

    def f(self, s):
        return s if s > 0 else self.k * s

    def d(self, y):
        return 1 if y > 0 else self.k
