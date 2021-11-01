import pickle
import numpy as np
from typing import Optional
import funsact


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


class EmptyAlphaError(Exception):
    def __str__(self):
        return """
    class Layer can't have empty alpha
    use special classes (LayerSigmoid, LayerLinear)
        """


class Layer:
    """Слой нейросети"""

    def __init__(
            self,
            lens: tuple[int, int],
            f_act=funsact.linear, d_f_act=funsact.d_linear):
        """
        - lens (количество нейронов этого и следующего слоя)
        - функции активации
        """
        self.w: np.ndarray = np.random.uniform(-0.5, 0.5, lens)
        self.t: np.ndarray = np.random.uniform(-0.5, 0.5, lens[1])
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

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error * self.d_f_act(y=self.y)
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error):
        raise EmptyAlphaError


class NeuralNetwork:
    def __init__(self, *args: Layer) -> None:
        """Создание нейросети с заданием массива слоев"""
        self.layers = args

    def go(self, x: np.ndarray) -> np.ndarray:
        """Прохождение всех слоев нейросети"""
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: np.ndarray, e: np.ndarray, alpha: Optional[float] = None) -> float:
        """Обучение набором обучающей выборки

        - x: (n, len) ... [[1 2] [2 3]]
        - e: (len,) ....... [3 4]
        """

        square_error = 0
        for i in range(len(e)):
            y = self.go(x[i])
            error = y - e[i]
            square_error += error ** 2 / 2
            for layer in reversed(self.layers):
                error = layer.back_propagation(error, alpha)
        return square_error

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
        if ans[0] == 'y':
            if filename is None:
                filename = input('Имя файла (*.nn): ') + '.nn'
            filename = 'nn_files/' + filename
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print('Сохранено в', filename)
        else:
            print('Сохранение отклонено')

    @staticmethod
    def load(filename=None):
        if filename is None:
            filename = input('Имя файла (*.nn): ') + '.nn'
        filename = 'nn_files/' + filename
        with open(filename, 'rb') as file:
            new_nn: NeuralNetwork = pickle.load(file)
        return new_nn


class LayerSigmoid(Layer):
    """Слой нейросети с сигмоидной функцией активации"""

    def __init__(self, lens: tuple[int, int]):
        """Слой нейросети с сигмоидной функцией активации"""
        super().__init__(lens, f_act=funsact.sigmoid, d_f_act=funsact.d_sigmoid)

    def adaptive_alpha(self, delta) -> float:
        alpha = 4 * sum(delta ** 2 * self.d_f_act(self.y)) \
                / (1 + sum(self.y ** 2)) \
                / sum((delta * self.y * (1 - self.y)) ** 2)
        return alpha


class LayerLinear(Layer):
    """Слой нейросети с линейной функцией активации"""

    def __init__(self, lens: tuple[int, int]):
        """Слой нейросети с линейной функцией активации"""
        super().__init__(lens, funsact.linear, funsact.d_linear)

    def adaptive_alpha(self, delta) -> float:
        alpha = 1 / (1 + (self.x ** 2).sum())
        return alpha
