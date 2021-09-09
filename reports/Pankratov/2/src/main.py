import math
import random
from typing import Tuple, List, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt

MIN_ERROR = 1.0e-27  # Минимальная ошибка для остановки обучени
TRAINING_SPEED = 0.01  # Скорость обучения нейронной сети


@dataclass(frozen=True)
class TrainingResult:
    w: List[float]
    T: float
    data_for_drawing: Tuple[List[int], List[float]]
    epochs: int


def func(x: float, a: int, b: int, d: float) -> float:
    return a * math.sin(b * x) + d


def calculate_output(
    nn_inputs: int, w: Sequence[float], t: float, training_outputs: Sequence[float], offset: int
) -> float:
    output = 0
    for j in range(nn_inputs):
        output += w[j] * training_outputs[j + offset]
    return output - t


def training_nn(
    nn_inputs: int, training_outputs: Sequence[float], is_adaptive_training_speed: bool = False
) -> TrainingResult:
    w = [random.uniform(0, 1) for _ in range(nn_inputs)]  # Веса
    T = random.uniform(0.5, 1)  # Порог
    data_for_drawing = ([], [])  # Данные для рисования графика
    training_speed = TRAINING_SPEED

    error, iteration, epochs = 1, 0, 0

    while error > MIN_ERROR:
        error = 0

        for i in range(len(training_outputs) - nn_inputs):
            if is_adaptive_training_speed:
                # Получение адаптивного шага обучения, лаб 2 формула(10)
                temp = 0
                for j in range(nn_inputs):
                    temp += training_outputs[i + j] ** 2
                training_speed = 1 / (1 + temp)

            # Получение выходного значения нейронной сети, формула(1.2)
            output = calculate_output(nn_inputs, w, T, training_outputs, i)

            # Обновление весов нейронной сети, формула(1.7)
            for j in range(nn_inputs):
                ideal_output = training_outputs[i + nn_inputs]
                w[j] -= training_speed * (output - ideal_output) * training_outputs[i + j]

            # Обновление порога нейронной сети, формула(1.8)
            T += training_speed * (output - training_outputs[i + nn_inputs])

            # Обновление среднеквадратичной ошибки нейронной сети, формула(1.3)
            error += (output - training_outputs[i + nn_inputs]) ** 2

        error /= len(training_outputs) - nn_inputs

        epochs += 1
        iteration += 1

        data_for_drawing[0].append(iteration)
        data_for_drawing[1].append(error)

    return TrainingResult(w, T, data_for_drawing, epochs)


def draw_graph(x_data: List[int], y_data: List[float]) -> None:
    plt.plot(x_data, y_data)
    plt.ylabel('error')
    plt.xlabel('iteration')
    plt.show()


def print_nn_output(w: List[float], t: float, ideal_outputs: List[float], nn_inputs: int) -> None:
    print(f"{'Reference value':<22}{'Receive value':<23}{'Deviation'}")

    for i in range(len(ideal_outputs) - nn_inputs):
        output = calculate_output(nn_inputs, w, t, ideal_outputs, i)
        print(f"{ideal_outputs[i+nn_inputs]:<20}  {output: <21}  {(ideal_outputs[i+nn_inputs] - output)}")


def print_nn_work_report(
    training_result: TrainingResult, nn_inputs: int, training_outputs: List[float], testing_outputs: List[float]
) -> None:
    draw_graph(*training_result.data_for_drawing)

    print("Training result:")
    print(f"Training epochs: {training_result.epochs}")
    print(f"Weight: {training_result.w}, T: {training_result.T}")
    print_nn_output(training_result.w, training_result.T, training_outputs, nn_inputs)

    print("\nTesting result:")
    print_nn_output(training_result.w, training_result.T, testing_outputs, nn_inputs)


def main():
    a, b, d = 3, 7, 0.3
    nn_inputs = 5  # Количество входных значений(входов)

    training_values, testing_values = 30, 15  # Количество значений функции для обучения и прогнозирования
    step = 0.1  # Шаг табуляции функции

    training_outputs = [func(i * step, a, b, d) for i in range(training_values)]
    testing_outputs = [func(i * step, a, b, d) for i in range(training_values, training_values + testing_values)]
    try:
        training_result = training_nn(nn_inputs, training_outputs, False)
    except OverflowError:
        return print("The neural network is divergent, the learning speed is too fast, reduce it")

    print("Without adaptive training speed")
    print_nn_work_report(training_result, nn_inputs, training_outputs, testing_outputs)

    training_result = training_nn(nn_inputs, training_outputs, True)
    print("With adaptive training speed")
    print_nn_work_report(training_result, nn_inputs, training_outputs, testing_outputs)


if __name__ == "__main__":
    main()
