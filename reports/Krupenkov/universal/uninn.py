import pickle
import numpy as np
from typing import Optional
import funsact


def predict_set(begin, lenght, count, step, function=None) -> tuple[np.ndarray, np.ndarray]:
    """Набор обучающей выборки типа:

    x = [ [1 2 3] [2 3 4] [3 4 5] ]
    e = [4 5 6]\n"""
    # temp - полный массив от 1 до (count + lenght), из которого берутся все значения
    temp = np.arange(count + lenght, dtype=np.double)
    x = np.zeros(shape=(count, lenght), dtype=np.double)
    for i in range(count):
        x[i] = temp[i: lenght + i]
    e = temp[lenght: lenght + count].reshape(-1, 1)

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
            self,
            lens: tuple[int, int],
            f_act=funsact.linear, d_f_act=funsact.d_linear,
            w=None, t=None):
        """
        - lens (количество нейронов этого и следующего слоя)
        - функции активации
        """
        self.lens = lens
        self.w: np.ndarray = np.random.uniform(-0.5, 0.5, lens) if w is None else w
        self.t: np.ndarray = np.random.uniform(-0.5, 0.5, lens[1]) if t is None else t
        self.f_act = f_act
        self.d_f_act = d_f_act

    def go(self, x: np.ndarray) -> np.ndarray:
        """Прохождение слоя"""
        self.x: np.ndarray = x
        self.s: np.ndarray = np.dot(x, self.w) - self.t
        self.y: np.ndarray = self.f_act(self.s)
        return self.y

    def back_propagation(self, error: np.ndarray, alpha: Optional[float], is_first_layer=False) -> Optional[np.ndarray]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error * self.d_f_act(self.y), self.w.transpose()) if not is_first_layer else None

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error * self.d_f_act(self.y)
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error):
        if not hasattr(self, 'd_f_act_0'):
            self.d_f_act_0 = self.d_f_act(self.f_act(0))
        alpha = (error ** 2 * self.d_f_act(self.y).sum()) \
                / self.d_f_act_0 \
                / (1 + (self.y ** 2).sum()) \
                / ((error * self.d_f_act(self.y)) ** 2).sum()
        return alpha


class NeuralNetwork:
    def __init__(self, *args: Layer) -> None:
        """Создание нейросети с заданием массива слоев"""
        self.layers = args

    def __str__(self) -> str:
        layer_info = ()
        for layer in self.layers:
            layer_info += (layer.w, layer.t)
        return str(layer_info)

    def go(self, x: np.ndarray) -> np.ndarray:
        """Прохождение всех слоев нейросети"""
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: np.ndarray, e: np.ndarray, alpha: Optional[float] = None):
        """Обучение набором обучающей выборки

        - x: (n, len) ... [[1 2] [2 3]]
        - e: (len,) ....... [3 4]
        """

        square_error = 0
        for i in range(len(e)):
            y = self.go(x[i])
            error = y - e[i]
            square_error += error ** 2 / 2
            for layer in self.layers[:0:-1]:
                error = layer.back_propagation(error, alpha)
            self.layers[0].back_propagation(error, alpha, is_first_layer=True)
        return square_error

    def prediction_results_table(self, x: np.ndarray, e: np.ndarray) -> None:
        """Красивый вывод прогона тестирующей выборки"""
        print('                эталон        выходное значение                  разница         среднекв. ошибка')
        y = self.go(x)
        y = y.reshape(-1)
        e = e.reshape(-1)
        delta = y - e
        square_error = delta ** 2 / 2
        for i in range(len(e)):
            print(f'{e[i] : 22}{y[i]: 25}{delta[i] : 25}{square_error[i] : 25}')
        print(f'  Finally square error:{square_error[-1] : 24}')

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

    def adaptive_alpha(self, error) -> float:
        alpha = 4 * (error ** 2 * self.d_f_act(self.y).sum()) \
                / (1 + (self.y ** 2).sum()) \
                / ((error * self.y * (1 - self.y)) ** 2).sum()
        return alpha


class LayerLinear(Layer):
    """Слой нейросети с линейной функцией активации"""

    def __init__(self, lens: tuple[int, int]):
        """Слой нейросети с линейной функцией активации"""
        super().__init__(lens, funsact.linear, funsact.d_linear)

    def adaptive_alpha(self, error) -> float:
        alpha = 1 / (1 + (self.x ** 2).sum())
        return alpha
