import numpy as np


def sigmoid(s):  # sigmoid
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):  # sigmoid'
    return y * (1 - y)


def linear(s):
    return s


def d_linear(y):
    return 1


class Layer:
    def __init__(self, lens: tuple[int, int],
                 f_act=linear, d_f_act=d_linear):
        self.w = np.random.uniform(0, 0.5, lens)
        self.t = np.random.uniform(0, 0.5)
        self.f_act = f_act
        self.d_f_act = d_f_act

    def go(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.s: np.ndarray = np.dot(x, self.w) - self.t
        self.y = self.f_act(self.s)
        return self.y

    def back_propagation(self, error: np.ndarray, alpha=0.5):
        # print(self.w, error, self.d_f_act(self.s), sep='\n', end='\n\n')
        error_later = np.dot(self.w * self.d_f_act(self.s.transpose()), error)

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

    def learn(self, x: np.ndarray, e, alpha=0.1):
        result = self.go(x)
        delta = result - e
        for layer in reversed(self.layers):
            delta = layer.back_propagation(delta, alpha=0.1)


def main():
    l1 = Layer(lens=(2, 3), f_act=sigmoid, d_f_act=d_sigmoid)
    l2 = Layer(lens=(3, 1), f_act=linear, d_f_act=d_linear)
    nn = NeuralNetwork([l1, l2])
    x = np.array([[0, 0], [1, 0], [1, 0], [1, 1]], float)
    for i in range(1):
        nn.learn(x=np.array([0, 0], float), e=0.0)
        nn.learn(x=np.array([0, 1], float), e=1.0)
        nn.learn(x=np.array([1, 0], float), e=1.0)
        nn.learn(x=np.array([1, 1], float), e=0.0)
    for el in x:
        print(nn.go(el))


if __name__ == '__main__':
    main()
