from math import sin
import random

inp, E, alpha = 4, 0.000000000001, 0.05
W, T = [random.uniform(-1, 1) for _ in range(inp)], random.uniform(-1, 1)
step = 0.3


def func_value(x) -> float:
    a, b, d = 4, 7, 0.2
    return a * sin(b * x) - d


def value_data() -> list:
    start = 1
    data_size = 10
    line = []
    for _ in range(data_size):
        x = start
        vector = []
        for _ in range(inp):
            vector.append(func_value(x))
            x += step
        line.append([vector, func_value(x)])
        start += step
    return line


def adaptive_step(vector: list):
    global alpha
    alpha = 1 / (1 + sum([x ** 2 for x in vector]))


def change(error: float, vector: list) -> None:
    global T, W
    for index in range(inp):
        W[index] -= alpha * error * vector[index]
    T += alpha * error


def predict(vector: list) -> float:
    output = 0
    for ind, val in enumerate(vector):
        output += W[ind] * val
    output -= T
    return output


def education() -> None:
    f_error = E + 1
    value = value_data()
    ep = 0
    print('{0:7}{1:<30}'.format("Эпоха", "Ошибка"))
    while abs(f_error) > E:
        f_error = 0
        for val in value:
            error = predict(val[0]) - val[1]
            adaptive_step(val[0])
            change(error, val[0])
            f_error += abs(error)
        ep += 1
        print('{0:<7}{1:<30}'.format(ep, f_error))
    print()

if __name__ == '__main__':
    education()
    data = value_data()
    print('{0:<30}{1:<30}{2:<30}'.format("Предсказание", "Эталонное значение", "Ошибка"))
    for d in data:
        p = predict(d[0])
        print('{0:<30}{1:<30}{2:<30}'.format(p, d[1], p - d[1]))
