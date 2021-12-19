import math
import random
import numpy as np

TRAINING_STEP = 0.25


class Layer:
    def __init__(self, in_size, nl_size):
        self.__training_step = TRAINING_STEP
        self.__input_values, self.__output_values = np.array([]), np.array([])
        self.error, self.__weights = np.array([]), []
        self.__T = [random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(nl_size)]
        for _ in range(nl_size):
            self.__weights.append([random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(in_size)])
        self.__weights = np.array(self.__weights)

    def set_values(self, input_values):
        self.__input_values = np.array(input_values)
        S = self.__input_values.dot(self.__weights.T)
        self.__output_values = np.array([1. / (1. + np.exp(-s)) for s in S])
        return self.__output_values

    def __error_init(self, error: list):
        weights = np.array(self.__weights).T
        self.error = [sum([error[ind] * self.__output_values[ind] * (1 - self.__output_values[ind]) * w for ind, w in
                           enumerate(weight)]) for weight in weights]

    def weights_change(self, error: list):
        self.__error_init(error)
        for r, W in enumerate(self.__weights):
            for c in range(len(W)):
                dF = self.__output_values[r] * self.__input_values[c] * (1 - self.__output_values[r])
                self.__weights[r][c] -= self.__training_step * error[r] * dF

        for ind in range(len(self.__T)):
            dF = self.__output_values[ind] * (1 - self.__output_values[ind])
            self.__T[ind] += self.__training_step * error[ind] * dF


class LastLayer:
    def __init__(self, in_size, out_size):
        self.__training_step = TRAINING_STEP
        self.__result, self.__weights = [], []
        self.error, self.__input_values = [], []
        self.__T = [random.uniform(-1, 1) for _ in range(out_size)]
        self.__T = [random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(out_size)]
        for _ in range(out_size):
            self.__weights.append([random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(in_size)])
        self.__weights = np.array(self.__weights)

    def set_values(self, input_values):
        self.__input_values = np.array(input_values)
        self.__result = self.__input_values.dot(self.__weights.T) - self.__T
        return self.__result

    def __error_init(self, error: list):
        self.error = self.__weights.T.dot(error)

    def weights_change(self, error):
        self.__error_init(error)
        for r in range(len(self.__weights)):
            for c in range(len(self.__weights[r])):
                self.__weights[r][c] -= self.__training_step * error[r] * self.__input_values[c]
        for ind, e in enumerate(error):
            self.__T += self.__training_step * e
