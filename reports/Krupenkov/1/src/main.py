from math import sin
from random import uniform
from typing import List

# Гиперпараметры обучения
MIN_ERROR = 9.0e-28  # Минимальная ошибка для остановки обучения
TRAINING_SPEED = 2.0e-1  # Скорость обучения нейронной сети


# Функция по условию (Вариант 9)
def function(x: float) -> float:
    return sin(8 * x) + 0.3


def main() -> None:
    inputs_amount = 5  # Количество входов
    step = 0.1  # Шаг табуляции функции

    training_epoch = 30  # Количество значений функции для обучения
    training_outputs: List[float] = [function(i * step) for i in range(training_epoch + inputs_amount)]
    # список значений функции для обучения

    # testing_epoch = 15  # Количество значений функции для прогнозирования
    # testing_outputs: List[float] = [function(i * step) for i in range(training_epoch, training_epoch + testing_epoch)]
    # список значений функции для прогнозирования

    # Создание списка случайных весов и порога
    w: List[float] = [uniform(0, 1) for _ in range(inputs_amount)]
    T: float = uniform(0, 1)

    # Инициализация счетчика и ошибки
    iteration = 0
    error = MIN_ERROR
    while error >= MIN_ERROR:
        error = 0
        iteration += 1

        for epoch in range(training_epoch):
            # Вычисление выходного значения (Формула 1.2)
            output: float = 0
            for j in range(inputs_amount):
                output += w[j] * training_outputs[epoch + j]
            output -= T

            # Значение функции
            ideal_output = training_outputs[epoch + inputs_amount]
            # Отклонение от функции
            error_output = output - ideal_output

            # Обновление весов нейронной сети (Формула 1.7)
            for t in range(inputs_amount):
                w[t] -= TRAINING_SPEED * error_output * training_outputs[epoch + t]

            # Обновление порога нейронной сети (Формула 1.8)
            T += TRAINING_SPEED * error_output

            # Обновление среднеквадратичной ошибки нейронной сети( Формула 1.3)
            error += (error_output ** 2) / 2

            # Вывод результатов
            print(f'Iteration {iteration + 1}\tEpoch {epoch + 1}:\t{ideal_output}\t{output}\t{error_output}\t{error}')

    print('Нейронная сеть обучена')


if __name__ == "__main__":
    main()
