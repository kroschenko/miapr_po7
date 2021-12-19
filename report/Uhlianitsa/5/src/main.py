import random
import math



vector_3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector_5 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
vector_8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
all_vectors = [vector_3, vector_5, vector_8]
alpha = 0.2
MIN_ERROR = 1e-4
input_neuron_number = 6
hidden_neuron_number = 2
output_neuron_number = 1
weight_arr_i_j = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(input_neuron_number - 1)]
weight_arr_j_k = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(output_neuron_number)]
T_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_neuron_number)]
T_out = [random.uniform(-0.5, 0.5) for _ in range(output_neuron_number)]


def sigmoidFunction(S):
    return 1 / (1 + math.exp(-S))


def calculate_S_hidden(y):
    s_hidden = []
    for j in range(hidden_neuron_number):
        value = 0
        for i in range(input_neuron_number - 1):
            value += y[i] * weight_arr_i_j[i][j]
        value -= T_hidden[j]
        s_hidden.append(sigmoidFunction(value))
    return s_hidden


def calculate_S_out(y_hidden):
    s_out = []
    for j in range(output_neuron_number):
        value = 0
        for i in range(hidden_neuron_number):
            value += y_hidden[i] * weight_arr_j_k[j][i]
        value -= T_out[j]
        s_out.append(sigmoidFunction(value))
    return s_out


def Wjk_change(y_hidden, y_out, error):
    global T_out
    for j in range(output_neuron_number):
        for i in range(hidden_neuron_number):
            weight_arr_j_k[j][i] -= alpha * error[j] * y_out[j] * (1 - y_out[j]) * y_hidden[i]
        T_out[j] += error[j] * alpha * y_out[j] * (1 - y_out[j])


def Wij_change(y_hidden, hidden_error, y):
    for j in range(hidden_neuron_number):
        for i in range(input_neuron_number - 1):
            weight_arr_i_j[i][j] -= alpha * hidden_error[j] * y[i] * y_hidden[j] * (1 - y_hidden[j])
        T_hidden[j] += alpha * hidden_error[j] * y_hidden[j] * (1 - y_hidden[j])


def main():
    hidden_error = [0 for i in range(hidden_neuron_number)]
    reference = [0 for i in range(output_neuron_number)]
    error_arr = [0 for i in range(output_neuron_number)]

    iter = 40
    epoch = 0
    error = 1
    while error > MIN_ERROR:
        error = 0
        for N in range(output_neuron_number):
            reference[N] = 1
            for i in range(iter):
                y = all_vectors[N]
                y_hidden = calculate_S_hidden(y)
                y_out = calculate_S_out(y_hidden)
                for i in range(output_neuron_number):
                    error_arr[i] = y_out[i] - reference[i]
                for j in range(hidden_neuron_number):
                    for k in range(output_neuron_number):
                        hidden_error[j] += error_arr[k] * y_out[k] * (1 - y_out[k]) * weight_arr_j_k[k][j]
                Wjk_change(y_hidden, y_out, error_arr)
                Wij_change(y_hidden, hidden_error, y)
                error += error_arr[N] ** 2
        error /= 2

        epoch += 1

    print(f'Number epoch : {epoch}')
    for i in range(len(all_vectors)):
        input = all_vectors[i]
        print("Vector :", i + 1, end=" : ")
        for j in range(len(vector_3)):
            print(input[j], end='')
        print("\nResult : ", end='')
        hidden_early = calculate_S_hidden(input)
        values = calculate_S_out(hidden_early)
        print(values[0])


if __name__ == '__main__':
    main()