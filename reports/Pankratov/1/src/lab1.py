import math
import random
from typing import Tuple, List, Sequence

MIN_ERROR = 0.000000000000000000000000001  # Минимальная ошибка для остановки обучения
TRAINING_SPEED = 0.01  # Скорость обучения нейронной сети


def func(x: float, a: int, b: int, d: float) -> float:
    return a * math.sin(b * x) + d


def calculate_output(
    nn_inputs: int, w: Sequence[float], t: float, training_outputs: Sequence[float], offset: int
) -> float:
    output = 0
    for j in range(nn_inputs):
        output += w[j] * training_outputs[j + offset]
    return output - t


def training_nn(nn_inputs: int, training_epoch: int, training_outputs: Sequence[float]) -> Tuple[List[float], float]:
    w = [random.uniform(0, 1) for _ in range(nn_inputs)]  # Веса
    T = random.uniform(0.5, 1)  # Порог

    error = 1
    while error > MIN_ERROR:
        error = 0

        for i in range(training_epoch - nn_inputs):
            # Получение выходного значения нейронной сети, формула(1.2)
            output = calculate_output(nn_inputs, w, T, training_outputs, i)

            # Обновление весов нейронной сети, формула(1.7)
            for j in range(nn_inputs):
                ideal_output = training_outputs[i + nn_inputs]
                w[j] -= TRAINING_SPEED * (output - ideal_output) * training_outputs[i + j]

            # Обновление порога нейронной сети, формула(1.8)
            T += TRAINING_SPEED * (output - training_outputs[i + nn_inputs])

            # Обновление среднеквадратичной ошибки нейронной сети, формула(1.3)
            error += 0.5 * ((output - training_outputs[i + nn_inputs]) ** 2)

    return w, T


def main():
    a, b, d = 3, 7, 0.3
    nn_inputs = 5  # Количество входных значений(входов)

    training_epoch, testing_epoch = 30, 15  # Количество значений функции для обучения и прогнозирования
    step = 0.1  # Шаг табуляции функции

    training_outputs = [func(i * step, a, b, d) for i in range(training_epoch)]
    testing_outputs = [func(i * step, a, b, d) for i in range(training_epoch, training_epoch + testing_epoch)]
    try:
        w, T = training_nn(nn_inputs, training_epoch, training_outputs)
    except OverflowError:
        return print("Нейронная сеть расходящаеся, скорость обучения слишком большая, уменьшите ее")

    print("Training result:")
    print(f"Weight: {w}, T: {T}")
    print("{:<22}{:<23}{}".format("Reference value", "Receive value", "Deviation"))

    for i in range(training_epoch - nn_inputs):
        output = calculate_output(nn_inputs, w, T, training_outputs, i)
        print(f"{training_outputs[i+nn_inputs]:<20}  {output: <21}  {(training_outputs[i+nn_inputs] - output)}")

    print("\nTesting result:")

    for i in range(testing_epoch - nn_inputs):
        output = calculate_output(nn_inputs, w, T, testing_outputs, i)
        print(f"{testing_outputs[i+nn_inputs]:<20}  {output: <21}  {(testing_outputs[i+nn_inputs] - output)}")


if __name__ == "__main__":
    main()
