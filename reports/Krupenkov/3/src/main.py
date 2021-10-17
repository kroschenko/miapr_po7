from math import cos, sin, exp
from pprint import pprint
from random import uniform
from typing import List


def function(x):
    return 0.1 * cos(0.3 * x) + 0.08 * sin(0.3 * x)


def activation(s):  # sigmoid
    return 1 / (1 + exp(-s))


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

        self.wx = [[uniform(-1, 1) for _ in range(self.h_amount)] for _ in range(self.x_amount)]
        self.wh = [uniform(-1, 1) for _ in range(self.h_amount)]
        self.Th = uniform(-1, 1)
        self.Ty = uniform(-1, 1)

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

NICE = ([[0.6887242652103013, 0.644229609531722, -0.2373665127607033, 0.21456380907081776],
  [0.42215503366482204, 0.057790857181553884, -0.4131235916438172, 0.22612885191988455],
  [0.4811252076699581, -0.2033901754339638, 0.4307421647338048, -0.0127652919652124],
  [-0.8788581441941105, -0.8548656153975062, 0.409305670758755, 0.07919464614077217],
  [-0.8880505962393973, -0.9222733455334121, -0.5328845171499988, 0.006945083098172511],
  [-0.8375969558232043, -0.5766891281013534, -0.5764582371952546, 0.853795074067204],
  [0.7021116029447848, 0.4943804171330565, -0.019178091556131104, 0.09438491384245103],
  [0.8158196510852577, -0.1259886396538579, -0.7028836266261288, -0.1664118193583017],
  [-0.8437175159858629, -0.15102465054536743, -0.6995373572622424, -0.8669600524532388],
  [-0.2745853836638393, -0.35644756303861885, -0.2735328444835077, 0.8072523244267449]],
 0.27786284789843707, [0.5534488256256959, -0.8353370167962032, -0.3951415250866152, -0.7259021118479324],
 -0.33073534370821966)

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
            if cum < 0.00001:
                for_coping = nn.get_weights()
                print('For coping:')
                pprint(for_coping, width=120, compact=True)
                break
            del nn

    # nn = NeuralNetwork(x_amount, h_amount, divider, t_learning, speed, epoch_amount, t_testing + 100)
    # nn.set_weights(*NICE)
    # nn.testing()


if __name__ == '__main__':
    main()
