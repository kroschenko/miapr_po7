from math import sin
from random import uniform, seed
from typing import List, Tuple
import matplotlib.pyplot as plt

# Гиперпараметры обучения
MIN_SQUARE_ERROR = 1e-25  # Минимальная ошибка для остановки обучения
TRAINING_EPOCH_AMOUNT = 30  # Количество значений функции (эпох) для обучения
TESTING_EPOCH_AMOUNT = 15  # Количество значений функции (эпох) для прогнозирования
MAX_ITERATIONS_AMOUNT = 18  # Максимальное количество повторений обучения


# Функция по условию (Вариант 9)
def function(x: float) -> float:
    return sin(8 * x) + 0.3


# Вывод нс
def calculate_output(
    inputs_amount: int, w: List[float], training_outputs: List[float], epoch: int, T: float
) -> float:
    # Вычисление выходного значения (Формула 1.2)
    output: float = 0
    for j in range(inputs_amount):
        output += w[j] * training_outputs[epoch + j]

    return output - T


def draw_error_iteration(x_data: List[int], y_data: List[float]) -> None:
    plt.plot(x_data, y_data)
    plt.ylabel('error^2')
    plt.xlabel('frame')
    plt.show()


def draw_sin(x_data: List[float], y_data: List[float]) -> None:
    plt.plot(x_data, y_data)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()


def main() -> None:
    # seed(20)
    inputs_amount = 5  # Количество входов
    step = 0.1  # Шаг табуляции функции

    # Значения функции для обучения
    training_outputs: List[float] = [function(i * step) for i in range(TRAINING_EPOCH_AMOUNT + inputs_amount)]

    # Значения функции для прогнозирования
    testing_outputs: List[float] = [
        function(i * step) for i in
        range(TRAINING_EPOCH_AMOUNT, TRAINING_EPOCH_AMOUNT + TESTING_EPOCH_AMOUNT + inputs_amount)
    ]

    w: List[float] = [uniform(0, 1) for _ in range(inputs_amount)]
    T: float = uniform(0, 1)  # Порог
    drawing_data_error: Tuple[List[int], List[float]] = ([], [])
    drawing_data_sin: Tuple[List[float], List[float]] = ([], [])

    # __________ Начало __________
    try:
        iteration = 0
        square_error: float = MIN_SQUARE_ERROR  # Среднеквадратичная ошибка

        # __________ Обучение __________
        while square_error >= MIN_SQUARE_ERROR and iteration < MAX_ITERATIONS_AMOUNT:

            square_error_sum = 0
            iteration += 1

            for epoch in range(TRAINING_EPOCH_AMOUNT):
                output = calculate_output(inputs_amount, w, training_outputs, epoch, T)
                ideal_output: float = training_outputs[epoch + inputs_amount]  # Истинное значение функции
                error: float = output - ideal_output  # Отклонение от функции

                # Изменение скорости обучение
                training_speed = 1 / (1 + sum([y ** 2 for y in training_outputs[epoch:epoch + inputs_amount]]))
                print(training_speed)

                # Обновление весов и порога (Формула 1.7, 1.8)
                for t in range(inputs_amount):
                    w[t] -= training_speed * error * training_outputs[epoch + t]
                T += training_speed * error

                # Сумма среднеквадратичных ошибок (Фейк формула 1.3)
                square_error_sum += error ** 2

                drawing_data_error[0].append(iteration * MAX_ITERATIONS_AMOUNT + epoch)
                drawing_data_error[1].append(error ** 2)
                drawing_data_sin[0].append(step * epoch)
                drawing_data_sin[1].append(output)

                # print(f'Iteration {iteration:3}  Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                #       f'{error:24}  {error ** 2 if error else "            are the same":24}')

            square_error = square_error_sum / TRAINING_EPOCH_AMOUNT
            print(f'Iteration {iteration:3}  Square error: {square_error}')

        # __________ Результаты обучения __________
        print('\nНейронная сеть обучена, результаты:')
        print(f'w: {w}\n'
              f'T: {T}\n')
        print('Тестирование на новом участке:')
        square_error_sum = 0

        # __________ Тестирование __________
        for epoch in range(TESTING_EPOCH_AMOUNT):
            output = calculate_output(inputs_amount, w, testing_outputs, epoch, T)

            ideal_output: float = testing_outputs[epoch + inputs_amount]  # Истинное значение функции
            error: float = output - ideal_output  # Отклонение от функции
            square_error_sum += error ** 2  # Суммарная ошибка (Формула 1.3)

            drawing_data_sin[0].append(step * (epoch + TRAINING_EPOCH_AMOUNT))
            drawing_data_sin[1].append(output)

            print(f'Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                  f'{error:24}  {error ** 2 if error else "            are the same":24}')

        square_error = square_error_sum / TRAINING_EPOCH_AMOUNT
        print(f'Testing square error: {square_error}')
        print(f'Количество эпох: {iteration}')

    except OverflowError:
        print('Слишком большая скорость обучения, выход из программы')


if __name__ == "__main__":
    main()
