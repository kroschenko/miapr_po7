from math import sin
from random import uniform
from typing import List

# Гиперпараметры обучения
MIN_SQUARE_ERROR = 1e-25  # Минимальная ошибка для остановки обучения
TRAINING_SPEED = 1.0e-1  # Скорость обучения нейронной сети
TRAINING_EPOCH_AMOUNT = 300  # Количество значений функции (эпох) для обучения
TESTING_EPOCH_AMOUNT = 15  # Количество значений функции (эпох) для прогнозирования


# Функция по условию (Вариант 9)
def function(x: float) -> float:
    return sin(8 * x) + 0.3


def main() -> None:
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
    T: float = uniform(0, 1)  # Порог

    try:  # Проверка на расходимость
        iteration = 0  # Счетчик итераций
        square_error: float = MIN_SQUARE_ERROR  # Среднеквадратичная ошибка

        # Обучение
        while square_error >= MIN_SQUARE_ERROR and iteration < 100:
            square_error = 0
            iteration += 1

            for epoch in range(TRAINING_EPOCH_AMOUNT):
                # Вычисление выходного значения (Формула 1.2)
                output: float = 0
                for j in range(inputs_amount):
                    output += w[j] * training_outputs[epoch + j]
                output -= T

                ideal_output: float = training_outputs[epoch + inputs_amount]  # Истинное значение функции
                error: float = output - ideal_output  # Отклонение от функции

                # Обновление весов нейронной сети (Формула 1.7)
                for t in range(inputs_amount):
                    w[t] -= TRAINING_SPEED * error * training_outputs[epoch + t]

                # Обновление порога нейронной сети (Формула 1.8)
                T += TRAINING_SPEED * error

                # Обновление среднеквадратичной ошибки нейронной сети (Формула 1.3)
                square_error += (error ** 2) / 2

                # Вывод результатов
                print(f'Iteration {iteration:3}  Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                      f'{error:24}  {square_error if error else "       like the previous":24}')

        print('\nНейронная сеть обучена, результаты:')
        print(f'w: {w}\n'
              f'T: {T}\n')
        print('Тестирование на новом участке:')

        square_error = 0
        # Тестирование
        for epoch in range(TESTING_EPOCH_AMOUNT):
            # Вычисление выходного значения (Формула 1.2)
            output: float = 0
            for j in range(inputs_amount):
                output += w[j] * testing_outputs[epoch + j]
            output -= T

            ideal_output: float = testing_outputs[epoch + inputs_amount]  # Истинное значение функции
            error: float = output - ideal_output  # Отклонение от функции

            # Обновление среднеквадратичной ошибки нейронной сети (Формула 1.3)
            square_error += (error ** 2) / 2

            # Вывод результатов
            print(f'Epoch {epoch + 1:2}:  {ideal_output:21}  {output:21}  '
                  f'{error:24}  {square_error if error else "       like the previous":24}')

    except OverflowError:
        print('Слишком большая скорость обучения, выход из программы')


if __name__ == "__main__":
    main()
