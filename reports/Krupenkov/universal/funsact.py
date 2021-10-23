import numpy as np


def sigmoid(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):  # sigmoid'
    """sigmoid'(s) = sigmoid(s) * (1 - sigmoid(s)) = y * (1 - y)"""
    return y * (1 - y)


def linear(s):
    return s


def d_linear(y):
    return 1.0
