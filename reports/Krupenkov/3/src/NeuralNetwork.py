import numpy as np
from pprint import pprint


def sigmoid(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):  # sigmoid'
    return y * (1 - y)


def linear(s):
    return s


def d_linear(y):
    return 1


class Layer:
    def __init__(self, weigths: np.ndarray, threshold: float = 0.5, f_act=linear, d_f_act=d_linear):
        self.w = weigths
        self.t = threshold
        self.f_act = f_act
        self.d_f_act = d_f_act

    def go(self, x: np.ndarray) -> np.ndarray:
        self.s = np.dot(x, self.w)
        y = self.f_act(self.s)
        return y

    def error_propagation(self, error: np.ndarray):
        self.error = error * self.w * self.d_f_act(self.s)
        print(error, self.w, self.d_f_act(self.s), self.error)
        return self.error



class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        # self.layer_count: int = len(layers)
        self.layers: list[Layer] = layers

    def go(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: np.ndarray, e: float):
        result = self.go(x)
        delta: float = result - e

        delta_array = np.array([delta], float)
        for layer in reversed(self.layers):
            delta_array = layer.error_propagation(delta_array)




def main():
    layer_input = Layer(weigths=np.array([[0.1, 0.2], [0.3, 0.4]]))
    layer_hidden = Layer(weigths=np.array([0.5, 0.6]))
    x = np.array([1., 2.])


    nn = NeuralNetwork([layer_input, layer_hidden])
    # print(nn.go(x))
    nn.learn(x, 1.)


if __name__ == '__main__':
    main()
