import math
import random
import numpy as np

TRAINING_STEP = 0.2


class Layer:
    def __init__(self, in_size, nl_size):
        self.__training_step = TRAINING_STEP
        self.__input_values = []
        self.output_values = []
        self.__weights = []
        self.error = []
        self.__T = [random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(nl_size)]
        for _ in range(nl_size):
            self.__weights.append([random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(in_size)])

    def set_values(self, input_values):
        self.__input_values = input_values
        self.output_values = [1 / (1 + math.exp(-s)) for s in self.__get_sum()]

    def __get_sum(self):
        S = []
        for r, W in enumerate(self.__weights):
            S.append(sum([w * self.__input_values[c] for c, w in enumerate(W)]) - self.__T[r])
        return S

    def __error_init(self, error: list):
        weights = np.array(self.__weights).T.tolist()
        self.error = [sum([error[ind] * self.output_values[ind] * (1 - self.output_values[ind]) * w for ind, w in
                           enumerate(weight)]) for weight in weights]

    def __adaptive_step(self):
        numerator = 4 * sum(
            [self.error[ind] ** 2 * self.__input_values[ind] * (1 - self.__input_values[ind]) for ind in
             range(len(self.error))])
        denominator = (1 + sum([y ** 2 for y in self.output_values])) * (sum(
            [(self.error[ind] * self.__input_values[ind] * (1 - self.__input_values[ind])) ** 2 for ind in
             range(len(self.error))]))
        self.__training_step = numerator / denominator

    def weights_change(self, error: list):
        self.__error_init(error)
        self.__adaptive_step()
        for r, W in enumerate(self.__weights):
            for c, w in enumerate(W):
                self.__weights[r][c] -= self.__training_step * error[r] * self.output_values[r] * \
                                        self.__input_values[c] * (1 - self.output_values[r])

        for ind in range(len(self.__T)):
            self.__T[ind] += self.__training_step * error[ind] * self.output_values[ind] * (1 - self.output_values[ind])


class LastLayer:
    def __init__(self, in_size):
        self.__training_step = TRAINING_STEP
        self.result = None
        self.error = []
        self.__input_values = []
        self.__weights = [random.uniform(-math.sqrt(in_size), math.sqrt(in_size)) for _ in range(in_size)]
        self.__T = random.uniform(-1, 1)

    def set_values(self, input_values):
        self.__input_values = input_values
        self.result = sum([w * self.__input_values[ind] for ind, w in enumerate(self.__weights)]) - self.__T

    def __error_init(self, error: list):
        self.error = [error * self.result * (1 - self.result) * w for w in self.__weights]

    def __adaptive_step(self):
        self.__training_step = 1 / (1 + sum([x ** 2 for x in self.__input_values]))

    def weights_change(self, error):
        self.__error_init(error)
        self.__adaptive_step()
        for ind in range(len(self.__weights)):
            self.__weights[ind] -= self.__training_step * error * self.__input_values[ind]
        self.__T += self.__training_step * error
