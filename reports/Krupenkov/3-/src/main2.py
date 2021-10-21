from pprint import pprint

import numpy as np


def function(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def activation(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_activation(y):  # sigmoid'
    return y * (1 - y)


class NeuralNetwork:
    def __init__(self, x_amount, h_amount, divider, t_max, speed, epoch_amount, t_testing):
        self.x_amount = x_amount
        self.h_amount = h_amount
        self.divider = divider
        self.t_learning = t_max
        self.speed = speed
        self.epoch_amount = epoch_amount
        self.t_testing = t_testing

        self.wx = np.random.uniform(-0.1, 0.1, (self.x_amount, self.h_amount))
        self.wh = np.random.uniform(-0.1, 0.1, self.h_amount)
        self.Th = np.random.uniform(-0.1, 0.1)
        self.Ty = np.random.uniform(-0.1, 0.1)

    def calculating(self, x):
        self.sh = 0
        self.h = []
        for j in range(self.h_amount):
            for i, xi in enumerate(x):
                self.sh += self.wx[i][j] * xi
            self.sh -= self.Th
            self.h.append(activation(self.sh))
        self.sy = 0
        for j in range(self.h_amount):
            self.sy += self.h[j] * self.wh[j]
        self.y = self.sy - self.Ty

    def learning(self):
        self.x = [i / self.divider for i in range(self.t_learning + self.x_amount)]
        self.e = self.x[self.x_amount:] + [(self.t_learning + self.x_amount) / self.divider]
        self.x = [function(x) for x in self.x]
        self.e = [function(y) for y in self.e]

        print(' N:      эталонное значение     полученное значение                 разница')

        for epoch in range(self.epoch_amount):
            print(f'Epoch: {epoch}')
            for t in range(self.t_learning):
                self.calculating(self.x[t:t + self.x_amount])

                self.y_delta = self.y - self.e[t]
                print(f'{t:2}: {self.e[t]:23} {self.y:23} {self.y_delta:23}')

                self.h_delta = 0
                for j in range(self.h_amount):
                    self.h_delta += self.y_delta * self.wh[j]

                for j in range(self.h_amount):
                    self.wh[j] -= self.speed * self.y_delta * self.y
                    self.Ty += self.speed * self.y_delta

                for j in range(self.h_amount):
                    for i in range(self.x_amount):
                        self.wx[i][j] -= self.speed * self.h_delta * d_activation(self.h[j]) * self.h[j]
                        self.Th += self.speed * self.h_delta * d_activation(self.y)

        print('Тестирование на новом участке с использованием предсказанных значений')
        print(' N:      эталонное значение     полученное значение                 разница')

    def testing(self):
        square_error = 0

        for t in range(self.t_testing):
            self.calculating(self.x)
            self.y_delta = self.y - self.e[t]
            square_error += self.y_delta ** 2
            print(f'{t:2}: {self.e[t]:23} {self.y:23} {self.y_delta:23}')
            self.x = self.x[1:] + [self.y]

        return square_error


def main():
    x_amount = 10
    h_amount = 4
    divider = 10
    t_learning = 30
    t_testing = 15
    speed = 0.2
    epoch_amount = 1

    x = np.array([[]], float)
    for i in range(t_learning + x_amount):
        x = np.append(x, np.array([np.arange(i, i + x_amount) / 10]))
    # x.reshape(t_learning, x_amount)
    e = np.arange(t_learning + x_amount, t_learning + x_amount + t_testing) / 10
    x = function(x)
    e = function(e)

    nn = NeuralNetwork(x_amount, h_amount, divider, t_learning, speed, epoch_amount, t_testing)
    nn.learning()
    nn.testing()


if __name__ == '__main__':
    main()
