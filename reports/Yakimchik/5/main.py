import random
import math
import matplotlib.pyplot as plt

STEP = 0.5
MIN_ERROR = 1e-4
input_neuron_number = 8
hidden_neuron_number = 6
output_neuron_number = 1
Wij = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(input_neuron_number - 1)]
Wjk = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(output_neuron_number)]
Tj = [random.uniform(-0.5, 0.5) for _ in range(hidden_neuron_number)]
Tk = [random.uniform(-0.5, 0.5) for _ in range(output_neuron_number)]

vector_1 = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
vector_2 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
vector_3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector_4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
vector_6 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
vector_8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
all_vectors = [vector_1, vector_2, vector_3, vector_4, vector_6, vector_8]


def sigmoidFunction(S):
    return 1 / (1 + math.exp(-S))


def Sj_onHiddenLayer(y):
    Sj = []
    for j in range(hidden_neuron_number):
        value = 0
        for i in range(input_neuron_number - 1):
            value += y[i] * Wij[i][j]
        value -= Tj[j]
        Sj.append(sigmoidFunction(value))
    return Sj


def Sk_onOutputLayer(Yj):
    Sk = []
    for j in range(output_neuron_number):
        value = 0
        for i in range(hidden_neuron_number):
            value += Yj[i] * Wjk[j][i]
        value -= Tk[j]
        Sk.append(sigmoidFunction(value))
    return Sk


def Wjk_change(Yj, Yk, error):
    global Tk
    for j in range(output_neuron_number):
        for i in range(hidden_neuron_number):
            Wjk[j][i] -= STEP * error[j] * Yk[j] * (1 - Yk[j]) * Yj[i]
        Tk[j] += error[j] * STEP * Yk[j] * (1 - Yk[j])


def Wij_change(Yj, hiddenLayer_error, y):
    for j in range(hidden_neuron_number):
        for i in range(input_neuron_number - 1):
            Wij[i][j] -= STEP * hiddenLayer_error[j] * y[i] * Yj[j] * (1 - Yj[j])
        Tj[j] += STEP * hiddenLayer_error[j] * Yj[j] * (1 - Yj[j])


def main():
    drawing_data = ([], [])
    errors = [0] * output_neuron_number
    reference = [0] * output_neuron_number
    hiddenLayer_error = [0] * hidden_neuron_number
    iteration = 1
    epoch = 0
    error = 1
    while error > MIN_ERROR:
        error = 0
        for N in range(output_neuron_number):
            reference[N] = 1
            for i in range(iteration):
                y = all_vectors[N]
                Yj = Sj_onHiddenLayer(y)
                Yk = Sk_onOutputLayer(Yj)
                for index in range(output_neuron_number):
                    errors[index] = Yk[index] - reference[index]
                for j in range(hidden_neuron_number):
                    for k in range(output_neuron_number):
                        hiddenLayer_error[j] += errors[k] * Yk[k] * (1 - Yk[k]) * Wjk[k][j]
                Wjk_change(Yj, Yk, errors)
                Wij_change(Yj, hiddenLayer_error, y)
                error += errors[N] ** 2
        error /= 2

        drawing_data[0].append(epoch)
        drawing_data[1].append(error)
        epoch += 1

    plt.plot(*drawing_data)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

    for i in range(len(all_vectors)):
        input = all_vectors[i]
        print("Result vector :", i + 1, end=" : ")
        for j in range(len(vector_1)):
            print(input[j], end='')
        print("\nResult : ", end='')
        hiddenLayer_prev = Sj_onHiddenLayer(input)
        Values = Sk_onOutputLayer(hiddenLayer_prev)
        print(Values[0])


if __name__ == '__main__':
    main()
