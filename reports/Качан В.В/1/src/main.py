import math
import random
import matplotlib.pyplot as plt

min_error = 1e-6
t_speed = 0.01


def func(x: float, a: int, b: int, d: float) -> float:
    return a * math.sin(b * x) + d


def f_output(weights: list[float], t: float, inputs_number: int, t_output: list[float], shift: int) -> float:
    output = 0
    for i in range(inputs_number):  # Столько значений будет подаваться на вход
        output += weights[i] * t_output[i + shift]  # Сдвиг для прогнозирования
    return output - t


def training(inputs_number: int, t_number: int, t_output: list[float]):
    weights = [random.uniform(-1, 1) for _ in range(inputs_number)]
    t = random.uniform(0.1, 1)
    data_for_drawing = ([], [])
    error = 1
    iteration = 0
    while error > min_error:
        error = 0
        iteration += 1
        for i in range(t_number - inputs_number):
            output_ = f_output(weights, t, inputs_number, t_output, i)
            # Изменение весовых коэффициентов
            for j in range(inputs_number):
                ideal_output = t_output[i + inputs_number]
                weights[j] -= t_speed * (output_ - ideal_output) * t_output[i + j]
            # Изменяем порог нейронной сети
            t += t_speed * (output_ - ideal_output)
            # Изменяем среднеквадратичную ошибку
            error += (output_ - ideal_output) ** 2
            error /= 2
        data_for_drawing[0].append(iteration)
        data_for_drawing[1].append(error)
    return weights, t, data_for_drawing


def main():
    a = 1
    b = 9
    d = 0.5
    inputs_number = 4
    t_number = 34
    test_number = 19
    step = 0.1
    t_output = [func(i * step, a, b, d) for i in range(t_number + inputs_number)]
    test_output = [func(i * step, a, b, d) for i in range(t_number, t_number + test_number)]

    training_weight, training_t, data_for_drawing = training(inputs_number, t_number, t_output)

    plt.plot(*data_for_drawing)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    print('Результаты обучения:')
    print(f'Веса: {training_weight}, Предел: {training_t}')
    print('{:<25}{:<27}{}'.format('Эталонное значение', 'Текущее значение', 'Погрешность'))

    for i in range(t_number - inputs_number):
        output_ = f_output(training_weight, training_t, inputs_number, t_output, i)
        print(f'{t_output[i + inputs_number]:<25} {output_:<25} {(t_output[i + inputs_number] - output_)}')

    print('\n\nРезультаты тестирования:')
    for i in range(test_number - inputs_number):
        output_ = f_output(training_weight, training_t, inputs_number, test_output, i)
        print(f'{test_output[i + inputs_number]:<25} {output_:<25} {(test_output[i + inputs_number] - output_)}')


if __name__ == "__main__":
    main()

