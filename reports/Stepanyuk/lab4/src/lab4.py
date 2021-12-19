# Вариант 9

import numpy as np


def f(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def predict_sets(begin, lenght, count, step):
    base_array = np.arange(count + lenght)
    input = np.zeros((count, lenght))
    for i in range(count):
        input[i] = base_array[i: lenght + i]
    etalons = base_array[lenght: lenght + count]
    input = input * step + begin
    etalons = etalons * step + begin
    return f(input), f(etalons)


class NeuralNet:
    def __init__(self):
        self.weights_x = np.random.normal(-1, 1, (10, 4))
        self.threshold_h = np.random.normal(-1, 1, 4)
        self.weights_h = np.random.normal(-1, 1, (4, 1))
        self.threshold_y = np.random.normal(-1, 1, 1)

    def go(self, x):
        self.input = x
        self.sum_h = np.dot(self.input, self.weights_x) - self.threshold_h
        self.hidden = 1.0 / (1.0 + np.exp(-self.sum_h))
        self.sum_y = np.dot(self.hidden, self.weights_h) - self.threshold_y
        self.output = self.sum_y
        return self.output

    def changing(self, error_y):
        error_h = np.dot(error_y, self.weights_h.transpose())
        alpha = 1 / (1 + np.square(self.hidden).sum())
        gamma_y = alpha * error_y
        self.weights_h -= np.dot(self.hidden.reshape(-1, 1), gamma_y.reshape(1, -1))
        self.threshold_y += gamma_y
        alpha = 4 * (error_h ** 2 * self.hidden * (1 - self.hidden)).sum() / (1 + (self.input ** 2).sum()) / np.square(error_h * self.hidden * (1 - self.hidden)).sum()
        gamma_h = alpha * error_h * self.output * (1.0 - self.output)
        self.weights_x -= np.dot(self.input.reshape(-1, 1), gamma_h.reshape(1, -1))
        self.threshold_h += gamma_h

    def learning(self, x, e):
        square_error = 0
        for i in range(len(e)):
            output = self.go(x[i])
            error = output - e[i]
            square_error += error ** 2 / 2
            self.changing(error)
        return square_error

    def testing(self, input, etalons):
        print(" эталонное значение       выходное значение    среднеквадратичная ошибочка")
        for i in range(len(etalons)):
            output = self.go(input[i])[0]
            delta = output - etalons[i]
            square_error = delta ** 2 / 2
            print(f"{etalons[i]: 19.17f}{output: 24.17f}{square_error: 26}")


neural_net = NeuralNet()
learn_input, learn_etalons = predict_sets(0, 10, 30, 0.1)
for i in range(30000):
    error = neural_net.learning(learn_input, learn_etalons)
    if i % 1000 == 0:
        print(f"{i}: {error}")
        if np.isnan(error):
            neural_net = NeuralNet()
test_input, test_etalons = predict_sets(3, 10, 15, 0.1)
neural_net.testing(test_input, test_etalons)
