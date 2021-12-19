import random
from math import sin, cos, exp


def ref(x):
    a, b, c, d = 0.1, 0.1, 0.05, 0.1
    return a * cos(b * x) + c * sin(d * x)


def calc_S_hid(y_hid_arr, w_hid_arr, ref_arr, T_hid_arr, input_nn, hidden_nn, iter):
    for i in range(len(y_hid_arr)):
        y_hid_arr[i] = 0

    for i in range(hidden_nn):
        for j in range(input_nn):
            y_hid_arr[i] += w_hid_arr[i][j] * ref_arr[iter + j]
        y_hid_arr[i] -= T_hid_arr[i]
    return y_hid_arr


def calc_y_hid(S_hid):
    return 1 / 1 + exp(-S_hid)


def calc_y(y_hid_arr, w_out_arr, T_out, hidden_nn):
    y = 0
    for i in range(hidden_nn):
        y += y_hid_arr[i] * w_out_arr[i]
    return y - T_out


def change_w_out(w_out_arr, gamma_out, y_hid_arr, hidden_nn):
    alpha = 0.1
    for i in range(hidden_nn):
        w_out_arr[i] -= alpha * gamma_out * y_hid_arr[i]
    return w_out_arr


def change_T_out(T_out, gamma_out):
    alpha = 0.1
    T_out += alpha * gamma_out
    return T_out


def change_w_hid(
    w_hid_arr, gamma_hidden, y_hid_arr, ref_arr, input_nn, hidden_nn, iter
):
    alpha = 0.1
    for i in range(hidden_nn):
        for j in range(input_nn):
            w_hid_arr[i][j] -= (
                alpha
                * gamma_hidden[i]
                * y_hid_arr[i]
                * (1 - y_hid_arr[i])
                * ref_arr[j + iter]
            )
    return w_hid_arr


def change_T_hid(T_hid_arr, gamma_hidden, y_hid_arr, hidden_nn):
    alpha = 0.1
    for i in range(hidden_nn):
        T_hid_arr[i] += alpha * gamma_hidden[i] * y_hid_arr[i] * (1 - y_hid_arr[i])
    return T_hid_arr


def main():
    step = 0.5
    input_nn = 6
    hidden_nn = 2
    train_value = 30
    test_value = 15
    ref_arr = [ref(x * step) for x in range(train_value)]

    w_hid_arr = []
    for i in range(hidden_nn):
        row = []
        for j in range(input_nn):
            row.append(random.uniform(0, 1))
        w_hid_arr.append(row)
    print(w_hid_arr)
    T_hid_arr = [0 for i in range(hidden_nn)]
    y_hid_arr = [0 for i in range(hidden_nn)]

    w_out_arr = [random.uniform(0, 1) for i in range(hidden_nn)]
    T_out = random.uniform(0, 1)

    gamma_out = 0
    gamma_hidden = [0 for i in range(hidden_nn)]

    epoch_counter = 0
    y = 0
    err = 1
    min_err = 1e-6
    while err > min_err:
        err = 0
        for iter in range(train_value - input_nn):

            y_hid_arr = calc_S_hid(
                y_hid_arr, w_hid_arr, ref_arr, T_hid_arr, input_nn, hidden_nn, iter
            )
            for i in range(hidden_nn):
                y_hid_arr[i] = calc_y_hid(y_hid_arr[i])

            y = calc_y(y_hid_arr, w_out_arr, T_out, hidden_nn)

            gamma_out = y - ref_arr[iter + input_nn]

            for i in range(hidden_nn):
                gamma_hidden[i] = gamma_out * y * (1 - y) * w_out_arr[i]

            w_out_arr = change_w_out(w_out_arr, gamma_out, y_hid_arr, hidden_nn)
            T_out = change_T_out(T_out, gamma_out)
            w_hid_arr = change_w_hid(
                w_hid_arr, gamma_hidden, y_hid_arr, ref_arr, input_nn, hidden_nn, iter
            )
            T_hid_arr = change_T_hid(T_hid_arr, gamma_hidden, y_hid_arr, hidden_nn)

            err += ((y - ref_arr[iter + input_nn]) ** 2) / 2
        epoch_counter += 1
        err /= train_value - input_nn
        if epoch_counter % 5000 == 0:
            print(err)

    print("Training end\nTraining result")
    print("{:^25}{:^25}{:^25}".format("Ref value", "Getting value", "Deviation"))
    for iter in range(train_value - input_nn):

        y_hid_arr = calc_S_hid(
            y_hid_arr, w_hid_arr, ref_arr, T_hid_arr, input_nn, hidden_nn, iter
        )
        for i in range(hidden_nn):
            y_hid_arr[i] = calc_y_hid(y_hid_arr[i])

        y = calc_y(y_hid_arr, w_out_arr, T_out, hidden_nn)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                ref_arr[iter + input_nn],
                y,
                ref_arr[iter + input_nn] - y,
            )
        )

    ref_test_arr = [
        ref(x * step) for x in range(train_value - input_nn, test_value + train_value)
    ]
    print("Testing result")
    print("{:^25}{:^25}{:^25}".format("Ref value", "Getting value", "Deviation"))
    for iter in range(test_value):

        y_hid_arr = calc_S_hid(
            y_hid_arr, w_hid_arr, ref_test_arr, T_hid_arr, input_nn, hidden_nn, iter
        )
        for i in range(hidden_nn):
            y_hid_arr[i] = calc_y_hid(y_hid_arr[i])

        y = calc_y(y_hid_arr, w_out_arr, T_out, hidden_nn)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                ref_test_arr[iter + input_nn],
                y,
                ref_test_arr[iter + input_nn] - y,
            )
        )


main()
