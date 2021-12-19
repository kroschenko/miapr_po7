
from math import sin, cos, exp
import random


input_neuron = 6
hidden_neuron = 2

min_error = 1e-9
alpha = 0.1
step = 0.1

weight_arr_i_j = [
    [random.uniform(-1, 1) for _ in range(hidden_neuron)]
    for _ in range(input_neuron - 1)
]
weight_arr_j_k = [random.uniform(-1, 1) for _ in range(hidden_neuron)]


def func(x):
    a, b, c, d = 0.2, 0.4, 0.09, 0.4
    return a * cos(b * x * step) + c * sin(d * x * step)


def ideal_value(x):
    y = []
    for i in range(input_neuron):
        y.append(func(x))
    return y


def calculate_S_hidden(j, y, T_hidden):
    S_hidden = 0
    for i in range(input_neuron - 1):
        S_hidden += y[i] * weight_arr_i_j[i][j]
    S_hidden -= T_hidden[j]
    return S_hidden


def calculate_S_out(y_hidden, T_out):
    S_out = 0
    for i in range(hidden_neuron):
        S_out += y_hidden[i] * weight_arr_j_k[i]
    S_out -= T_out
    return S_out


def sigmoid_function(j, y, T_hidden):
    S_hidden = calculate_S_hidden(j, y, T_hidden)
    return 1 / (1 + exp(-S_hidden))


def Wjk_change(y_hidden, error,step_k):
    for i in range(hidden_neuron):
        weight_arr_j_k[i] -= step_k * error * y_hidden[i]


def Wij_change(y_hidden, error, y,step_j):
    for i in range(input_neuron - 1):
        for j in range(hidden_neuron):
            weight_arr_i_j[i][j] -= (
                    step_j
                    * error
                    * weight_arr_j_k[j]
                    * y[i]
                    * (y_hidden[j] * (1 - y_hidden[j]))
            )

def calculate_T_hidden(T_hidden, y_hidden, error, step_hidden):
    for i in range(hidden_neuron):
        T_hidden[i] += step_hidden * error * y_hidden[i] * (1 - y_hidden[i])


def calculate_T_out(T_out, local_error, step_out):
    T_out += local_error * step_out

def adaptive_step(y_hidden, y_out, error):
    num, comp_1 = 1, 1
    error_j = []
    for i in range(hidden_neuron):
        error_j.append(error * y_out * (1 - y_out) * weight_arr_j_k[i])
    for i in range(hidden_neuron):
        num += (error_j[i] ** 2) * y_hidden[i] * (1 - y_hidden[i])
        comp_1 += (error_j[i] ** 2) * (y_hidden[i] ** 2) * ((1 - y_hidden[i]) ** 2)
    step_j = (4 * num) / ((1 + y_out ** 2) * comp_1)
    return step_j


def main():
    epoch = 0
    sum_error = 1
    T_hidden = [0 for _ in range(hidden_neuron)]
    T_out = 0
    step_hidden = 0
    step_out = 0

    while sum_error > min_error:
        for k in range(30):
            y_hidden = []
            y_out, sum_error = 1, 0
            y_ideal = ideal_value(k)
            for i in range(hidden_neuron):
                y_hidden.append(sigmoid_function(i, y_ideal, T_hidden))
            y_out = calculate_S_out(y_hidden, T_out)
            local_error = y_out - y_ideal[0]
            step_hidden = adaptive_step(y_hidden, y_out, local_error)
            step_out = 1 / (1 + (y_hidden[0] ** 2) + (y_hidden[1] ** 2))
            Wjk_change(y_hidden, local_error,step_out)
            Wij_change(y_hidden, local_error, y_ideal,step_hidden)
            calculate_T_hidden(T_hidden, y_hidden, local_error,step_hidden)
            calculate_T_out(T_out, local_error,step_out)
            sum_error += 0.5 * (local_error ** 2)
        epoch += 1
        print(f'Number epoch:{epoch}:{local_error}')
    print(f"Number of epoch: {epoch}")
    print("Testing result:")
    print(
        "{:<30}{:<30}{}".format("Reference value", "Current value", "Error")
    )
    for i in range(30, 45):
        y_hidden_test = []
        y_out_test, sum_error_ = 0, 0
        y_ideal_test = ideal_value(i)
        for j in range(hidden_neuron):
            y_hidden_test.append(sigmoid_function(j, y_ideal_test, T_hidden))
        y_out_test = calculate_S_out(y_hidden_test, T_out)
        local_error_ = y_out_test - y_ideal_test[-1]
        sum_error_ += 0.5 * (local_error_ ** 2)
        calculate_T_hidden(T_hidden, y_hidden_test, local_error,step_hidden)
        calculate_T_out(T_out, local_error,step_out)

        print(f"{y_ideal_test[-1]:<29} {y_out_test:<29} {local_error_}")


if __name__ == "__main__":
    main()
