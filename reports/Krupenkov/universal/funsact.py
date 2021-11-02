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


def relu(s):
    return s if s > 0 else 0


def d_relu(y):
    return 1 if y > 0 else 0


def parametric_relu(s, k=0.01):
    return s if s > 0 else k * s


def d_parametric_relu(y, k=0.01):
    return 1 if y > 0 else k
