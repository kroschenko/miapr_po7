import pickle
from typing import Optional

import funsact
import numpy as np


def predict_set(begin, lenght, count, step, function=None) -> tuple[np.ndarray, np.ndarray]:
    """Набор обучающей выборки типа:

    x = [ [1 2 3] [2 3 4] [3 4 5] ]
    e = [4 5 6]\n"""
    temp = np.arange(count + lenght, dtype=np.double)
    x = np.zeros(shape=(count, lenght), dtype=np.double)
    for i in range(count):
        x[i] = temp[i: lenght + i]
    e = temp[lenght: lenght + count]

    x = x * step + begin
    e = e * step + begin
    if function is None:
        return x, e
    else:
        return function(x), function(e)


def shuffle_set(x, e) -> tuple[np.ndarray, np.ndarray]:
    """Перемешивание набора обучающей выборки"""
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    e = e[randomize]
    return x, e


class Layer:
    """Слой нейросети"""

    def __init__(
            self, lens: tuple[int, int],
            f_act=funsact.linear, d_f_act=funsact.d_linear):
        """
        - lens (количество нейронов этого и следующего слоя)
        - функции активации
        """
        self.w: np.ndarray = np.random.uniform(-0.5, 0.5, lens)
        self.t: float = np.random.uniform(-0.5, 0.5)
        self.f_act = f_act
        self.d_f_act = d_f_act

    def go(self, x: np.ndarray) -> np.ndarray:
        """Прохождение слоя"""
        self.x: np.ndarray = x
        self.s: np.ndarray = np.dot(x, self.w) - self.t
        self.y: np.ndarray = self.f_act(self.s)
        return self.y

    def back_propagation(self, error: np.ndarray, alpha: Optional[float]) -> np.ndarray:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error * self.d_f_act(y=self.y), self.w.transpose())
        if alpha is None:
            # alpha = sum(error ** 2 * self.d_f_act(self.y)) \
            #         / self.d_f_act(self.f_act(0)) \
            #         / (1 + sum(error ** 2 * self.d_f_act(self.y) ** 2))
            print(f'''{sum(error ** 2 * self.d_f_act(self.y))}
            {self.d_f_act(y=self.f_act(0))}
            {(1 + (self.y ** 2).sum())}
            {sum((self.y * self.d_f_act(self.y)) ** 2)}''')
            alpha = sum(error ** 2 * self.d_f_act(self.y)) \
                    / self.d_f_act(y=self.f_act(0)) \
                    / (1 + (self.y ** 2).sum()) \
                    / sum((self.y * self.d_f_act(self.y)) ** 2)
            print(f'alpha: {alpha}')

        for j in range(self.w.shape[1]):
            for i in range(self.w.shape[0]):
                gamma = alpha * error[j] * self.d_f_act(y=self.y[j])
                self.w[i][j] -= gamma * self.x[i]
                self.t += gamma

        return error_later


class NeuralNetwork:
    def __init__(self, *args: Layer) -> None:
        """Создание нейросети с заданием массива слоев"""
        self.layers = args

    def go(self, x: np.ndarray) -> np.ndarray:
        """Прохождение всех слоев нейросети"""
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: np.ndarray, e: np.ndarray, alpha: Optional[float] = None) -> None:
        """Обучение набором обучающей выборки

        - x.shape = (n, len)
        - e.shape(len,)
        """

        for i in range(len(e)):
            result = self.go(x[i])
            delta = result - e[i]
            for layer in reversed(self.layers):
                delta = layer.back_propagation(delta, alpha)

    def go_results(self, x, e) -> None:
        """Красивый вывод прогона тестирующей выборки"""
        print('                эталон        выходное значение                  разница         среднекв. ошибка')
        square_error = 0
        for i in range(len(e)):
            y: float = self.go(x[i])[0]
            delta = y - e[i]
            square_error += delta ** 2 / 2
            print(f'{e[i] : 22}{y: 25}{delta : 25}{square_error : 25}')

    def save(self, filename=None) -> None:
        ans = input('Желаете сохранить? (y/n): ')
        if ans == 'y':
            if filename is None:
                filename = input('Имя файла (*.nn): ') + '.nn'
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print('Сохранено в', filename)
        else:
            print('Сохранение отклонено')

    @staticmethod
    def load(filename=None):
        if filename is None:
            filename = input('Имя файла (*.nn): ') + '.nn'
        with open(filename, 'rb') as file:
            new_nn: NeuralNetwork = pickle.load(file)
        return new_nn
