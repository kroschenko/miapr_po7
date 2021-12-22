from math import sin, cos, exp
import random


input_neuron = 6
hidden_neuron = 2

min_error = 1e-9
alpha = 0.1
step = 0.1

weight_i_j = [
    [random.uniform(-1, 1) for _ in range(hidden_neuron)]
    for _ in range(input_neuron - 1)
]
weight_arr_j_k = [random.uniform(-1, 1) for _ in range(hidden_neuron)]


def func(x):
    a, b, c, d = 0.2, 0.4, 0.09, 0.4
    return a * cos(b * x * step) + c * sin(d * x * step)


def idl_valye(x):
    y = []
    for i in range(input_neuron):
        y.append(func(x))
    return y


def calculate_hidden(j, y, T_hidden):
    S_hidden = 0
    for i in range(input_neuron - 1):
        S_hidden += y[i] * weight_i_j[i][j]
    S_hidden -= T_hidden[j]
    return S_hidden


def calculate_out(y_hidden, T_out):
    S_out = 0
    for i in range(hidden_neuron):
        S_out += y_hidden[i] * weight_arr_j_k[i]
    S_out -= T_out
    return S_out


def sigmoid(j, y, T_hidden):
    S_hidden = calculate_hidden(j, y, T_hidden)
    return 1 / (1 + exp(-S_hidden))


def Wjk_change(y_hidden, error):
    for i in range(hidden_neuron):
        weight_arr_j_k[i] -= alpha * error * y_hidden[i]


def Wij_change(y_hidden, error, y):
    for i in range(input_neuron - 1):
        for j in range(hidden_neuron):
            weight_i_j[i][j] -= (
                    alpha
                    * error
                    * weight_arr_j_k[j]
                    * y[i]
                    * (y_hidden[j] * (1 - y_hidden[j]))
            )

def calculate_T_hidden(T_hidden, y_hidden, error):
    for i in range(hidden_neuron):
        T_hidden[i] += alpha * error * y_hidden[i] * (1 - y_hidden[i])


def calculate_T_out(T_out, local_error):
    T_out += local_error * alpha


def main():
    epoch = 0
    sum_error = 1
    T_hidden = [0 for _ in range(hidden_neuron)]
    T_out= 0
    while sum_error > min_error:
        for k in range(30):
            y_hidden = []
            y_out, sum_error = 0, 0
            y_ideal = idl_valye(k)
            for i in range(hidden_neuron):
                y_hidden.append(sigmoid(i, y_ideal, T_hidden))
            y_out = calculate_out(y_hidden, T_out)
            local_error = y_out - y_ideal[0]
            Wjk_change(y_hidden, local_error)
            Wij_change(y_hidden, local_error, y_ideal)
            calculate_T_hidden(T_hidden, y_hidden, local_error)
            calculate_T_out(T_out, local_error)
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
        y_ideal_test = idl_valye(i)
        for j in range(hidden_neuron):
            y_hidden_test.append(sigmoid(j, y_ideal_test, T_hidden))
        y_out_test = calculate_out(y_hidden_test, T_out)
        local_error_ = y_out_test - y_ideal_test[-1]
        sum_error_ += 0.5 * (local_error_ ** 2)
        calculate_T_hidden(T_hidden, y_hidden_test, local_error)
        calculate_T_out(T_out, local_error)

        print(f"{y_ideal_test[-1]:<29} {y_out_test:<29} {local_error_}")


if __name__ == "__main__":
    main()
