import math
import random
import matplotlib.pyplot as plt


def etalonn(x):
    a, b, d = 1, 5, 0.1  # parametres for calculate reference values
    return a * math.sin(b * x) + d


def getting_value(w_arr, etalon_arr, T, i, amount_inputs) -> float:
    value = 0
    for j in range(amount_inputs):
        value += w_arr[j] * etalon_arr[j + i]
    return value - T


def calc_alpha(etalon_arr, i, amount_inputs) -> float:
    alpha = 0
    for j in range(amount_inputs):
        alpha += etalon_arr[j + i] ** 2
    return (1 / (1 + alpha))


def calc_w(w_arr, alpha, etalon_arr, value, i, amount_inputs) -> list:
    for j in range(len(w_arr)):
        w_arr[j] -= alpha * etalon_arr[i + j] * (value - etalon_arr[i + amount_inputs])
    return w_arr


def calc_T(T, alpha, etalon_arr, value, i, amount_inputs) -> float:
    T += alpha * (value - etalon_arr[i + amount_inputs])
    return T


def main():
    amount_inputs = 3  # amount of input neuron
    alpha = 0
    min_error = 1.0e-30
    step = 0.1  # tabulation step
    amount_train = 30  # number training input
    amount_test = 15  # number testing input
    w_arr = [
        random.uniform(0, 1) for i in range(amount_inputs)
    ]  # generate weights in array
    T = random.uniform(0, 1)  # generate limit
    etalon_arr = [
        etalonn(i * step) for i in range(amount_train)
    ]  # reference values for training
    etalon_for_testing = [  # reference values for testing
        etalonn(i * step) for i in range(amount_train, amount_test + amount_train)
    ]
    arr_chart = []  # array for build chart
    error = 10
    epoch_iteration = 0
    while error > min_error:
        error = 0
        for i in range(amount_train - amount_inputs):
            value = getting_value(w_arr, etalon_arr, T, i, amount_inputs)
            alpha = calc_alpha(etalon_arr, i, amount_inputs)
            w_arr = calc_w(w_arr, alpha, etalon_arr, value, i, amount_inputs)
            T = calc_T(T, alpha, etalon_arr, value, i, amount_inputs)
            error += (value - etalon_arr[i + amount_inputs]) ** 2
        error /= amount_train - amount_inputs
        epoch_iteration += 1
        arr_chart.append(error)

    print("Training end\nTraining result")
    print("{:^25}{:^25}{:^25}".format("Etalon value", "Getting value", "Deviation"))
    for i in range(amount_train - amount_inputs):
        value = getting_value(w_arr, etalon_arr, T, i, amount_inputs)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                etalon_arr[i + amount_inputs],
                value,
                etalon_arr[i + amount_inputs] - value,
            )
        )
    print("Testing result:", epoch_iteration, "epoch")
    print("{:^25}{:^25}{:^25}".format("Etalon value", "Getting value", "Deviation"))
    for i in range(amount_test - amount_inputs):
        value = getting_value(w_arr, etalon_for_testing, T, i, amount_inputs)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                etalon_for_testing[i + amount_inputs],
                value,
                etalon_for_testing[i + amount_inputs] - value,
            )
        )
    plt.plot(arr_chart)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


main()
