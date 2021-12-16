import math
import random
import matplotlib.pyplot as plt
from typing import List
min_error = 1e-6


def func(x: float, a: int, b: int, d: float) -> float:
    return a * math.sin(b * x) + d


def speed_training(t_output: List[float], i: int, input_number) -> float:
    speed_training = 0
    for j in range(input_number):
        speed_training += t_output[i + j]**2
    return 1 / (1 + speed_training)


def my_output(
    weights: List[float],
    t: float,
    input_number: int,
    t_output: List[float],
    shift: int,
) -> float:
    output_ = 0
    for i in range(input_number):  # Столько значений будет подаваться на вход
        output_ += weights[i] * t_output[i + shift]  # Сдвиг для прогнозирования
    return output_ - t


def training(input_number: int, t_number: int, t_output: List[float]):
    weights = [random.uniform(-1, 1) for _ in range(input_number)]
    t = random.uniform(0.1, 1)
    data_for_drawing = ([], [])
    error = 1
    iter = 0
    while error > min_error:
        error = 0
        iter += 1
        for i in range(t_number - input_number):
            output = my_output(weights, t, input_number, t_output, i)
            training_speed = speed_training(t_output, i, input_number)
            # Изменение весовых коэффициентов
            for j in range(input_number):
                ideal_output = t_output[i + input_number]
                weights[j] -= (
                    training_speed * (output - ideal_output) * t_output[i + j]
                )
            # Изменяем порог нейронной сети
            t += training_speed * (output - ideal_output)
            # Изменяем среднеквадратичную ошибку
            error += (output - ideal_output) ** 2
            error /= 2
        data_for_drawing[0].append(iter)
        data_for_drawing[1].append(error)
    return weights, t, data_for_drawing, iter


def main():
    a = 1
    b = 9
    d = 0.5
    input_number = 4
    t_number = 34
    test_number = 19
    step = 0.1
    t_output = [
        func(i * step, a, b, d) for i in range(t_number + input_number)
    ]
    test_output = [
        func(i * step, a, b, d)
        for i in range(t_number, t_number + test_number)
    ]

    training_weight, training_t, data_for_drawing, epochs = training(input_number, t_number, t_output)

    plt.plot(*data_for_drawing)
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.show()

    print("Результаты обучения:")
    print(f"Веса: {training_weight}, Предел: {training_t}")
    print(
        "{:<25}{:<27}{}".format("Эталонное значение", "Текущее значение", "Погрешность")
    )

    for i in range(t_number - input_number):
        output_ = my_output(
            training_weight, training_t, input_number, t_output, i
        )
        print(
            f"{t_output[i + input_number]:<25} {output_:<25} {(t_output[i + input_number] - output_)}"
        )
    print(f'Number of epochs: {epochs}')

    print("\n\nРезультаты тестирования:")
    for i in range(test_number - input_number):
        output_ = my_output(training_weight, training_t, input_number, test_output, i)
        print(
            f"{test_output[i + input_number]:<25} {output_:<25} {(test_output[i + input_number] - output_)}"
        )


if __name__ == "__main__":
    main()
