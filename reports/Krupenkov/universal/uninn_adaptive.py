from uninn import *


def d_sigmoid(y):  # sigmoid'
    return sigmoid(y) * (1 - sigmoid(y))


def back_propagation(self, error: np.ndarray, alpha):
    error_later = np.dot(error * self.d_f_act(self.s), self.w.transpose())
    alpha = sum(error ** 2 * self.d_f_act(self.s)) \
            / self.d_f_act(0) \
            / (1 + sum(error ** 2)) / (sum(self.d_f_act(self.s)))

    for j in range(self.w.shape[1]):
        for i in range(self.w.shape[0]):
            gamma = alpha * error[j] * self.d_f_act(self.s[j])
            self.w[i][j] -= gamma * self.x[i]
            self.t += gamma

    return error_later


Layer.back_propagation = back_propagation
