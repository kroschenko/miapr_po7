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
    return (s > 0) * s


def d_relu_0(y):
    return (y > 0) * 1


class Relu:
    def __init__(self, k):
        """ReLu с параметром k"""
        self.k = k

    def f(self, s):
        return s * self.k + (s > 0) * s * (1 - self.k)
        # return s if s > 0 else self.k * s

    def d(self, y):
        return self.k + (y > 0) * (1 - self.k)
        # return 1 if y > 0 else self.k
