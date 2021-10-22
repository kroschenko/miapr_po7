import pickle
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


def shuffle_set(x, e):
    """Перемешивание набора обучающей выборки"""
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    e = e[randomize]
    return x, e


def sigmoid(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):  # sigmoid'
    return y * (1 - y)


def linear(s):
    return s


def d_linear(y):
    return 1


class Layer:
    """Слой нейросети

    Параметры: слын
    """
    def __init__(self, lens: tuple[int, int],
                 f_act=linear, d_f_act=d_linear):
        self.w: np.ndarray = np.random.uniform(-0.5, 0.5, lens)
        self.t: float = np.random.uniform(-0.5, 0.5)
        self.f_act = f_act
        self.d_f_act = d_f_act

    def go(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.s: np.ndarray = np.dot(x, self.w) - self.t
        self.y = self.f_act(self.s)
        return self.y

    def back_propagation(self, error: np.ndarray, alpha):
        error_later = np.dot(error * self.d_f_act(self.s), self.w.transpose())

        for j in range(self.w.shape[1]):
            for i in range(self.w.shape[0]):
                gamma = alpha * error[j] * self.d_f_act(self.s[j])
                self.w[i][j] -= gamma * self.x[i]
                self.t += gamma

        return error_later


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def go(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: np.ndarray, e: float, alpha=0.1):
        result = self.go(x)
        delta = result - e
        for layer in reversed(self.layers):
            delta = layer.back_propagation(delta, alpha)

    def go_results(self, x, e):
        print('                эталон        выходное значение                  разница         среднекв. ошибка')
        for i in range(len(e)):
            y: float = self.go(x[i])[0]
            print(f'{e[i] : 22}{y: 25}{abs(e[i] - y) : 25}{(e[i] - y) ** 2 : 25}')

    def save(self, filename=None):
        ans = input('Желаете сохранить? (y/n): ')
        if ans == 'y':
            if filename is None:
                filename = input('Имя файла (*.nn): ') + '.nn'
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print('Сохранено в', filename)
        else:
            print('Сохранение отклонено')
        return

    @staticmethod
    def load(filename=None):
        if filename is None:
            filename = input('Имя файла (*.nn): ') + '.nn'
        with open(filename, 'rb') as file:
            new_nn = pickle.load(file)
        return new_nn
