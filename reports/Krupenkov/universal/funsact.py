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
