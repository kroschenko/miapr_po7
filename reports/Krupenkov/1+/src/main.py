from math import sin
from random import uniform, seed
from typing import List

# Гиперпараметры обучения
MIN_SQUARE_ERROR = 1e-25  # Минимальная ошибка для остановки обучения
TRAINING_SPEED = 1.0e-1  # Скорость обучения нейронной сети
TRAINING_EPOCH_AMOUNT = 30  # Количество значений функции (эпох) для обучения
TESTING_EPOCH_AMOUNT = 15  # Количество значений функции (эпох) для прогнозирования
MAX_ITERATIONS_AMOUNT = 20


# Функция по условию (Вариант 9)
def function(x: float) -> float:
    return sin(8 * x) + 0.3


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

    w: List[float] = [uniform(0, 1) for _ in range(inputs_amount)]  # Список всех весов
    t: float = uniform(0, 1)  # Порог

    try:  # Проверка на расходимость
        iteration = 0  # Счетчик итераций
        square_error: float = MIN_SQUARE_ERROR  # Среднеквадратичная ошибка

        # Обучение
        while square_error >= MIN_SQUARE_ERROR and iteration < MAX_ITERATIONS_AMOUNT:
            square_error_sum = 0
            iteration += 1

            for epoch in range(TRAINING_EPOCH_AMOUNT):
                # Вычисление выходного значения (Формула 1.2)
                output: float = 0
                for j in range(inputs_amount):
                    output += w[j] * training_outputs[epoch + j]
                output -= t

                ideal_output: float = training_outputs[epoch + inputs_amount]  # Истинное значение функции
                error: float = output - ideal_output  # Отклонение от функции

                # Обновление весов нейронной сети (Формула 1.7)
                for i in range(inputs_amount):
                    w[i] -= TRAINING_SPEED * error * training_outputs[epoch + i]

                # Обновление порога нейронной сети (Формула 1.8)
                t += TRAINING_SPEED * error

                # Обновление среднеквадратичной ошибки нейронной сети (Формула 1.3)
                square_error_sum += error ** 2

                # Вывод результатов
                # print(f'Iteration {iteration:3}  Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                #       f'{error:24}  {error ** 2 if error else "            are the same":24}')

            square_error = square_error_sum / TRAINING_EPOCH_AMOUNT
            # print(f'Iteration {iteration:3}  Square error: {square_error}')

        print('\nНейронная сеть обучена, результаты:')
        print(f'w: {w}\n'
              f'T: {t}\n')
        print('Тестирование на новом участке:')
        print('Epoch  N:     Идеальное значение    Полученное значение          '
              'Локальная ошибка       Квадратичная ошибка   ')

        square_error_sum = 0
        # Тестирование
        for epoch in range(TESTING_EPOCH_AMOUNT):
            # Вычисление выходного значения (Формула 1.2)
            output: float = 0
            for j in range(inputs_amount):
                output += w[j] * testing_outputs[epoch + j]
            output -= t

            ideal_output: float = testing_outputs[epoch + inputs_amount]  # Истинное значение функции
            error: float = output - ideal_output  # Отклонение от функции

            # Обновление среднеквадратичной ошибки нейронной сети (Формула 1.3)
            square_error_sum += error ** 2

            # Вывод результатов
            print(f'Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                  f'{error:24}  {error ** 2 if error else "            are the same":24}')

        square_error = square_error_sum / TRAINING_EPOCH_AMOUNT
        print(f'Среднеквадратичная ошибка: {square_error}')
        print(f'Количество эпох: {iteration}')

    except OverflowError:
        print('Слишком большая скорость обучения, выход из программы')


if __name__ == "__main__":
    main()
