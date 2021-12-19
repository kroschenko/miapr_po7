import math
import random
import matplotlib.pyplot as plt


def func(x):
    a, b, d = 2, 9, 0.4
    return a * math.sin(b * x) + d


def output_value(
    weight_arr, reference_arr, T, shift, amount_inputs
) -> float:  # Формула 1.2
    # Формула 1.2 нахождение выходного значения
    value = 0
    for j in range(amount_inputs):
        value += weight_arr[j] * reference_arr[j + shift]
    return value - T


def calc_w(weight_arr, t_alpha, reference_arr, value, shift, amount_inputs, T):
    # Формула 1.7 обновление весов
    for j in range(len(weight_arr)):
        weight_arr[j] -= (
            t_alpha
            * (value - reference_arr[amount_inputs + shift])
            * reference_arr[shift + j]
        )


def main():
    train_epoch = 30  # количество значений для обучения
    prediction_epoch = 15  # количество значений для прогнозирования
    amount_inputs = 3  # количество входных нейронов
    t_alpha = 0.01  # скорость обучения
    step = 0.1  # шаг табулирования
    min_error = 1.0e-27
    weight_arr = [
        random.uniform(0, 1) for i in range(amount_inputs)
    ]  # массив рандомных весов
    T = random.uniform(0, 1)  # порог
    reference_arr = [
        func(i * step) for i in range(train_epoch)
    ]  # эталонные значения для обучения
    error = 1
    plot_arr = []

    while error > min_error:
        error = 0
        for i in range(train_epoch - amount_inputs):
            value = output_value(weight_arr, reference_arr, T, i, amount_inputs)
            calc_w(weight_arr, t_alpha, reference_arr, value, i, amount_inputs, T)
            T += t_alpha * (
                value - reference_arr[amount_inputs + i]
            )  # формула 1.8 обновление порога
            error += (value - reference_arr[i + amount_inputs]) ** 2
        error /= train_epoch - amount_inputs
        plot_arr.append(error)
    plt.plot(plot_arr)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    print("Training results:")
    print(f"Weight arr:{weight_arr},T:{T}")
    print("{:30}{:30}{:30}".format("Reference value", "Output value", "Difference"))

    for i in range(train_epoch - amount_inputs):
        value = output_value(weight_arr, reference_arr, T, i, amount_inputs)
        print(
            "{:<30}{:<30}{:<30}".format(
                reference_arr[amount_inputs + i],
                value,
                reference_arr[amount_inputs + i] - value,
            )
        )
    reference_arr_t = [
        func(i * step) for i in range(train_epoch, train_epoch + prediction_epoch)
    ] # массив эталонных значений для прогнозирования

    print("Testing results:")
    print("{:30}{:30}{:30}".format("Reference value", "Output value", "Difference"))
    for i in range(prediction_epoch - amount_inputs):
        value = output_value(weight_arr, reference_arr_t, T, i, amount_inputs)
        print(
            "{:<30}{:<30}{:<30}".format(
                reference_arr_t[amount_inputs + i],
                value,
                reference_arr_t[amount_inputs + i] - value,
            )
        )


if __name__ == "__main__":
    main()
