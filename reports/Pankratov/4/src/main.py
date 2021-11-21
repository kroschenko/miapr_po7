from typing import Tuple, Callable, Any
from functools import partial

import numpy as np
from numpy.typing import NDArray

LEARNING_SPEED = 0.1

WeightsType = NDArray[NDArray[np.float64]]
TType = NDArray[np.float64]


def func(a: float, b: float, c: float, d: float, x) -> float:
    return a * np.cos(b * x) + c * np.sin(d * x)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def d_sigmoid(y):
    return y * (1 - y)


def lin(s):
    return s


def d_lin(_):
    return 1.0


def generate_dataset(
    start: int, stop: int, count: int, func: Callable[[Any], float], step: float
) -> Tuple[NDArray[NDArray[np.float64]], NDArray[NDArray[np.float64]]]:
    inputs, outputs = [], []
    for i in range(start, stop):
        inputs_sample = func(np.arange(i, count + i) * step)
        outputs_sample = np.array([func((count + i) * step)])

        inputs.append(inputs_sample)
        outputs.append(outputs_sample)

    return np.array(inputs), np.array(outputs)


def calc_error(
    outputs: NDArray[NDArray[np.float64]], ideal_outputs: NDArray[NDArray[np.float64]]
) -> NDArray[np.float64]:
    errors = (outputs - ideal_outputs) ** 2
    return errors.sum(axis=0) / len(ideal_outputs)


def calc_percent(
    outputs: NDArray[NDArray[np.float64]], ideal_outputs: NDArray[NDArray[np.float64]]
) -> NDArray[np.float64]:
    percents = (outputs * 100) / ideal_outputs
    return percents.sum(axis=0) / len(ideal_outputs)


def training(
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    w_output: WeightsType,
    w_hidden: WeightsType,
    T_output: TType,
    T_hidden: TType,
    is_adaptive_learning_speed: bool = False,
) -> None:
    hidden_outputs = sigmoid(np.dot(w_hidden, inputs) - T_hidden)
    output_outputs = lin(np.dot(w_output, hidden_outputs) - T_output)

    error_output = output_outputs - outputs
    error_hidden = error_output * d_sigmoid(hidden_outputs) * w_output

    learning_speed_output = LEARNING_SPEED
    learning_speed_hidden = LEARNING_SPEED
    if is_adaptive_learning_speed:
        ls_output_numerator = (np.square(error_output) * d_lin(hidden_outputs)).sum()
        ls_output_denominator = 1 + np.square(error_output * d_lin(hidden_outputs)).sum()
        learning_speed_output = ls_output_numerator / ls_output_denominator

        ls_hidden_numerator = 4 * (np.square(error_hidden) * d_sigmoid(hidden_outputs)).sum()
        ls_hidden_denominator = (1 + np.square(output_outputs).sum()) * np.square(error_hidden * d_sigmoid(hidden_outputs)).sum()
        learning_speed_hidden = ls_hidden_numerator / ls_hidden_denominator

    w_output -= learning_speed_output * np.dot(error_output.reshape(-1, 1), hidden_outputs.reshape(1, -1))
    T_output += learning_speed_output * error_output.reshape(-1)

    w_hidden -= learning_speed_hidden * np.dot(error_hidden.reshape(-1, 1), inputs.reshape(1, -1))
    T_hidden += learning_speed_hidden * error_hidden.reshape(-1)


def learn(
    inputs: NDArray[NDArray[np.float64]],
    ideal_outputs: NDArray[NDArray[np.float64]],
    w_output: WeightsType,
    w_hidden: WeightsType,
    T_output: TType,
    T_hidden: TType,
    epochs: int = 1000,
    is_print_intermediate_result: bool = True,
    is_adaptive_learning_speed: bool = False
):
    for epoch in range(epochs + 1):
        for input, output in zip(inputs, ideal_outputs):
            training(input, output, w_output, w_hidden, T_output, T_hidden, is_adaptive_learning_speed)

        if is_print_intermediate_result and not epoch % (epochs // 10):
            outputs = v_predict(inputs, w_hidden, w_output, T_hidden, T_output)
            errors = calc_error(outputs, ideal_outputs)
            percents = calc_percent(outputs, ideal_outputs)
            print(f"{epoch} error: {errors}, percent: {percents}")

    return w_output, w_hidden, T_output, T_hidden


def predict(
    inputs: NDArray[np.float64], w_hidden: WeightsType, w_output: WeightsType, T_hidden: TType, T_output: TType
) -> NDArray[np.float64]:
    hidden_outputs = sigmoid(np.dot(w_hidden, inputs) - T_hidden)
    return lin(np.dot(w_output, hidden_outputs) - T_output)


def print_results(epochs, inputs, ideal_outputs, w_output, w_hidden, T_output, T_hidden):
    outputs = v_predict(inputs, w_hidden, w_output, T_hidden, T_output)
    difference = ideal_outputs - outputs

    error = calc_error(outputs, ideal_outputs)
    percent = calc_percent(outputs, ideal_outputs)
    print(f"Epochs: {epochs}")
    print(f"Error: {error[0]}")
    print(f"Percent: {percent[0]}")
    print("Prediction:")
    print("         Ideal output:                  Output:              Difference:")
    for i in range(len(difference)):
        print(f"{ideal_outputs[i][0]: 22}{outputs[i][0]: 25}{difference[i][0]: 25}")


VPredictType = Callable[
    [NDArray[NDArray[np.float64]], WeightsType, WeightsType, TType, TType], NDArray[NDArray[np.float64]]
]
v_predict: VPredictType = np.vectorize(predict, signature="(n)->(m)", excluded=[1, 2, 3, 4])


def main():
    EPOCHS = 10_000

    A, B, C, D = 0.3, 0.3, 0.07, 0.3
    STEP = 0.1

    nn_inputs = 10
    nn_hidden = 4
    nn_output = 1

    training_values, testing_values = 30, 15

    wrapped_func = partial(func, A, B, C, D)
    inputs, ideal_outputs = generate_dataset(0, training_values, nn_inputs, wrapped_func, STEP)
    test_inputs, test_ideal_outputs = generate_dataset(
        training_values, training_values + testing_values, nn_inputs, wrapped_func, STEP
    )

    w_hidden: WeightsType = np.random.normal(-1, 1, (nn_hidden, nn_inputs))
    w_output: WeightsType = np.random.normal(-1, 1, (nn_output, nn_hidden))
    T_hidden: TType = np.random.normal(-1, 1, nn_hidden)
    T_output: TType = np.random.normal(-1, 1, nn_output)

    not_adaptive_result = learn(
        inputs, ideal_outputs, w_output.copy(), w_hidden.copy(), T_output.copy(), T_hidden.copy(), EPOCHS, False, False
    )
    print("No adaptive learning speed result:")
    print("Learning dataset:")
    print_results(EPOCHS, inputs, ideal_outputs, *not_adaptive_result)
    print()
    print("Testing dataset:")
    print_results(EPOCHS, test_inputs, test_ideal_outputs, *not_adaptive_result)

    adaptive_result = learn(
        inputs, ideal_outputs, w_output.copy(), w_hidden.copy(), T_output.copy(), T_hidden.copy(), EPOCHS, False, True
    )
    print()
    print("Adaptive learning speed result:")
    print("Learning dataset:")
    print_results(EPOCHS, inputs, ideal_outputs, *adaptive_result)
    print()
    print("Testing dataset:")
    print_results(EPOCHS, test_inputs, test_ideal_outputs, *adaptive_result)


if __name__ == "__main__":
    main()
