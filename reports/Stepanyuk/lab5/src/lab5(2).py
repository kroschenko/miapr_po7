# Вариант 9

import numpy as np


class NeuralNet:
    def __init__(self):
        self.weights_x = np.random.normal(-1, 1, (20, 4))
        self.threshold_h = np.random.normal(-1, 1, 4)
        self.weights_h = np.random.normal(-1, 1, (4, 3))
        self.threshold_y = np.random.normal(-1, 1, 3)

    def go(self, x):
        self.input = x
        self.sum_h = np.dot(self.input, self.weights_x) - self.threshold_h
        self.hidden = 1.0 / (1.0 + np.exp(-self.sum_h))
        self.sum_y = np.dot(self.hidden, self.weights_h) - self.threshold_y
        self.output = self.sum_y
        return self.output

    def changing(self, error_y, alpha):
        error_h = np.dot(error_y, self.weights_h.transpose())
        gamma_y = alpha * error_y
        self.weights_h -= np.dot(self.hidden.reshape(-1, 1), gamma_y.reshape(1, -1))
        self.threshold_y += gamma_y
        gamma_h = alpha * error_h * self.hidden * (1.0 - self.hidden)
        self.weights_x -= np.dot(self.input.reshape(-1, 1), gamma_h.reshape(1, -1))
        self.threshold_h += gamma_h

    def learning(self, x, e, alpha):
        square_error = 0
        for i in range(len(e)):
            output = self.go(x[i])
            error = output - e[i]
            square_error += (error ** 2 / 2).sum()
            self.changing(error, alpha)
        return square_error


vectors = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    ]
)
etalons = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)
neural_net = NeuralNet()
speed = 0.06
for i in range(30000):
    error = neural_net.learning(vectors, etalons, speed)
    if error <= 1e-3:
        break
for i in range(3):
    print(f"\nВектор {i}")
    output = neural_net.go(vectors[i])
    result = output.argmax() == i
    print(f" 20/20: {result}")
    for numb, j in enumerate(np.random.choice(20, 20, replace=False)):
        vectors[i][j] = 1 - vectors[i][j]
        output = neural_net.go(vectors[i])
        result = output.argmax() == i
        print(f"{19 - numb: 3}/20: {result}")
