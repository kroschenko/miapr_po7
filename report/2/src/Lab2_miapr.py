import math
import random
import matplotlib.pyplot as plt


def func(x):
    a, b, d = 2, 9, 0.4
    return a * math.sin(b * x) + d


def output(weight, reference, T, shift, number_inputs) -> float: 
    # Формула 1.2 нахождение выходного значения
    value = 0
    for j in range(number_inputs):
        value += weight[j] * reference[j + shift]
    return value - T


def calc(weight, learning_rate, reference, value, shift, number_inputs, T):
    # Формула 1.7 обновление весов
    for j in range(len(weight)):
        weight[j] -= (
            learning_rate
            * (value - reference[number_inputs + shift])
            * reference[shift + j]
        )
def adaptive_alpha(reference,number_inputs,shift):
    temp = 0
    for i in range(number_inputs):
        temp += reference[i+shift]**2
    return 1 / (1 + temp)



def main():
    training = 30  # количество значений для обучения
    prediction = 15  # количество значений для прогнозирования
    number_inputs = 3  # количество входных нейронов
    learning_rate = 0  # скорость обучения
    step = 0.1  # шаг табулирования
    min_еrror = 1.0e-29
    weight = [random.uniform(0, 1) for i in range(number_inputs)]  # массив рандомных весов
    T = random.uniform(0, 1)  # порог
    reference = [func(i * step) for i in range(training)]  # эталонные значения для обучения
    error = 1
    plot = []

    while error > min_еrror:
        error = 0
        for i in range(training - number_inputs):
            value = output(weight, reference, T, i, number_inputs)
            learning_rate = adaptive_alpha(reference, number_inputs, i)
            calc(weight, learning_rate, reference, value, i, number_inputs, T)
            T += learning_rate * (value - reference[number_inputs + i])  # формула 1.8 обновление порога
            error += (value - reference[i + number_inputs]) ** 2
        error /= training - number_inputs
        plot.append(error)
    plt.plot(plot)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    print("Training results:")
    print(f"Weight arr:{weight},T:{T}")
    print("{:30}{:30}{:30}".format("Reference value", "Output value", "Difference"))

    for i in range(training - number_inputs):
        value = output(weight, reference, T, i, number_inputs)
        print(
            "{:<30}{:<30}{:<30}".format(
                reference[number_inputs + i],
                value,
                reference[number_inputs + i] - value,
            )
        )
    reference_t = [func(i * step) for i in range(training, training + prediction)] # массив эталонных значений для прогнозирования

    print("Testing results:")
    print("{:30}{:30}{:30}".format("Reference value", "Output value", "Difference"))
    for i in range(prediction - number_inputs):
        value = output(weight, reference_t, T, i, number_inputs)
        print(
            "{:<30}{:<30}{:<30}".format(
                reference_t[number_inputs + i],
                value,
                reference_t[number_inputs + i] - value,
            )
        )


if __name__ == "__main__":
    main()
