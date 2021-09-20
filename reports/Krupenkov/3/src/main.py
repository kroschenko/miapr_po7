from math import cos, sin, exp
from random import uniform
from typing import List


def function(x):
    return 0.1 * cos(0.3 * x) + 0.08 * sin(0.3 * x)


def activation(s):  # sigmoid
    return 1 / (1 + exp(-s))


def d_activation(y):
    return y * (1 - y)


class NeuralNetwork:
    x_amount: int
    h_amount: int
    t_max: int
    speed: float
    epoch_amount: int

    x: List[float]
    e: List[float]
    wx: List[List[float]]
    Th: float
    h: List[float]
    sh: float
    wh: List[float]
    y: float
    sy: float

    def __init__(self, x_amount, h_amount, t_max, speed, epoch_amount):
        self.x_amount = x_amount
        self.h_amount = h_amount
        self.t_max = t_max
        self.speed = speed
        self.epoch_amount = epoch_amount

        self.x = [i / 10 for i in range(t_max + x_amount)]
        self.e = self.x[x_amount:] + [(t_max + x_amount) / 10]
        self.x = [function(x) for x in self.x]
        self.e = [function(y) for y in self.e]
        print(self.x)
        print(self.e)
        self.wx = [[uniform(-1, 1) for j in range(self.h_amount)] for i in range(self.x_amount)]
        self.wh = [uniform(-1, 1) for j in range(self.h_amount)]
        self.Th = uniform(-1, 1)
        self.Ty = uniform(-1, 1)

    def calculating(self, t):
        self.sh = 0
        self.h = []
        for j in range(self.h_amount):
            for i in range(self.x_amount):
                self.sh += self.wx[i][j] * self.x[i + t]
            self.sh -= self.Th
            self.h.append(activation(self.sh))
        self.sy = 0
        for j in range(self.h_amount):
            self.sy += self.h[j] * self.wh[j]
        self.y = self.sy - self.Ty

    def learning(self):
        for epoch in range(self.epoch_amount):
            for t in range(self.t_max):
                self.calculating(t)
                delta: float = self.y - self.e[t]
                print(self.e[t], self.y, delta)

                for j in range(self.h_amount):
                    self.wh[j] -= self.speed * delta * self.h[j]
                    self.Ty += self.speed * delta

                for j in range(self.h_amount):
                    for i in range(self.x_amount):
                        self.wx[i][j] -= self.speed * delta * d_activation(self.h[j]) * self.x[i]
                        self.Th += self.speed * delta * d_activation(self.y)


def main():
    x_amount = 10
    h_amount = 4
    t_max = 30
    speed = 0.2
    epoch_amount = 3

    nn = NeuralNetwork(x_amount, h_amount, t_max, speed, epoch_amount)
    nn.learning()


if __name__ == '__main__':
    main()
