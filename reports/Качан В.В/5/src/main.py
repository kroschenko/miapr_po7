import random
import math
import matplotlib.pyplot as plt

STEP = 0.5
MIN_ERROR = 1e-4
input_neuron = 8
hidden_neuron = 3
output_neuron = 1
Wij = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron)] for _ in range(input_neuron - 1)]
Wjk = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron)] for _ in range(output_neuron)]
Tj = [random.uniform(-0.5, 0.5) for _ in range(hidden_neuron)]
Tk = [random.uniform(-0.5, 0.5) for _ in range(output_neuron)]

v1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
v2 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
v3 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
v4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
v6 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
v8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
Vector = [v1, v2, v3, v4, v6, v8]


def sigmoid(S):
    return 1 / (1 + math.exp(-S))


def Sj_HiddenLayer(y):
    Sj = []
    for j in range(hidden_neuron):
        value = 0
        for i in range(input_neuron - 1):
            value += y[i] * Wij[i][j]
        value -= Tj[j]
        Sj.append(sigmoid(value))
    return Sj


def Sk_OutputLayer(Yj):
    Sk = []
    for j in range(output_neuron):
        value = 0
        for i in range(hidden_neuron):
            value += Yj[i] * Wjk[j][i]
        value -= Tk[j]
        Sk.append(sigmoid(value))
    return Sk


def Wjk_change(Yj, Yk, error):
    global Tk
    for j in range(output_neuron):
        for i in range(hidden_neuron):
            Wjk[j][i] -= STEP * error[j] * Yk[j] * (1 - Yk[j]) * Yj[i]
        Tk[j] += error[j] * STEP * Yk[j] * (1 - Yk[j])


def Wij_change(Yj, hiddenLayer_error, y):
    for j in range(hidden_neuron):
        for i in range(input_neuron - 1):
            Wij[i][j] -= STEP * hiddenLayer_error[j] * y[i] * Yj[j] * (1 - Yj[j])
        Tj[j] += STEP * hiddenLayer_error[j] * Yj[j] * (1 - Yj[j])


def main():
    drawing_data = ([], [])
    errors = [0] * output_neuron
    reference = [0] * output_neuron
    hiddenLayer_error = [0] * hidden_neuron
    iteration = 1
    epoch = 0
    error = 1
    while error > MIN_ERROR:
        error = 0
        for N in range(output_neuron):
            reference[N] = 1
            for i in range(iteration):
                y = Vector[N]
                Yj = Sj_HiddenLayer(y)
                Yk = Sk_OutputLayer(Yj)
                for index in range(output_neuron):
                    errors[index] = Yk[index] - reference[index]
                for j in range(hidden_neuron):
                    for k in range(output_neuron):
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

    for i in range(len(Vector)):
        input = Vector[i]
        print("Result vector :", i + 1, end=" : ")
        for j in range(len(v1)):
            print(input[j], end='')
        print("\nResult : ", end='')
        hiddenLayer_prev = Sj_HiddenLayer(input)
        Values = Sk_OutputLayer(hiddenLayer_prev)
        print(Values[0])


if __name__ == '__main__':
    main()