import numpy as np
from pprint import pprint
from random import uniform
from typing import List


def function(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def activation(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_activation(y):  # sigmoid'
    return y * (1 - y)


class NeuralNetwork:
    x_amount: int
    h_amount: int
    divider: float
    t_learning: int
    t_testing: int
    speed: float
    epoch_amount: int

    x: List[float]
    e: List[float]
    wx: List[List[float]]
    Th: float
    Ty: float
    h: List[float]
    sh: float
    wh: List[float]
    y: float
    sy: float
    y_delta: float
    h_delta: float

    def __init__(self, x_amount, h_amount, divider, t_max, speed, epoch_amount, t_testing):
        self.x_amount = x_amount
        self.h_amount = h_amount
        self.divider = divider
        self.t_learning = t_max
        self.speed = speed
        self.epoch_amount = epoch_amount
        self.t_testing = t_testing

        self.wx = [[uniform(-0.1, 0.1) for _ in range(self.h_amount)] for _ in range(self.x_amount)]
        self.wh = [uniform(-0.1, 0.1) for _ in range(self.h_amount)]
        # self.Th = uniform(-0.1, 0.1)
        # self.Ty = uniform(-0.1, 0.1)
        self.Th, self.Ty = 0, 0


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
        self.x = [i / self.divider for i in range(self.t_learning, self.t_learning + self.x_amount)]
        self.e = [i / self.divider for i in range(self.t_learning + self.x_amount,
                                                  self.t_learning + self.x_amount + self.t_testing)]
        self.x = [function(x) for x in self.x]
        self.e = [function(y) for y in self.e]
        square_error = 0

        for t in range(self.t_testing):
            self.calculating(self.x)
            self.y_delta = self.y - self.e[t]
            square_error += self.y_delta ** 2
            print(f'{t:2}: {self.e[t]:23} {self.y:23} {self.y_delta:23}')
            self.x = self.x[1:] + [self.y]

        return square_error

    def get_weights(self):
        print(f'\nwx: ')
        pprint(self.wx, width=120, compact=True)

        print(
            f'''Th:
{self.Th}
wh: 
{self.wh}
Ty: 
{self.Ty}
''')
        return self.wx, self.Th, self.wh, self.Ty

    def set_weights(self, wx, Th, wh, Ty):
        self.wx, self.Th, self.wh, self.Ty = wx, Th, wh, Ty


def main():
    x_amount = 10
    h_amount = 4
    divider = 10
    t_learning = 30
    t_testing = 15
    speed = 0.2
    epoch_amount = 1

    i = 0
    while True:
        nn = NeuralNetwork(x_amount, h_amount, divider, t_learning, speed, epoch_amount, t_testing)
        try:
            nn.learning()
        finally:
            cum = nn.testing()
            print(f'Новая нейронная сеть №{i}: {cum}')
            i += 1
            if cum < 0.0001:
                for_coping = nn.get_weights()
                print('For coping:')
                pprint(for_coping, width=120, compact=True)
                break
            del nn


if __name__ == '__main__':
    main()
