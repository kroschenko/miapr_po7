from random import random as r
from math import exp as exp

vectors = [
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
]
vectors_code = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
]
x_list = [vectors[0], vectors[5], vectors[7]]
e_list = [vectors_code[0], vectors_code[5], vectors_code[7]]
Wki = [[r() for i in range(20)], [r() for i in range(20)]]
Wij = [[r() for i in range(8)], [r() for i in range(8)]]
Ti = [r() for i in range(2)]
Tj = [r() for i in range(8)]
ls = 0.01
MSE = 0


def sigm(x):
    return 1 / (1 + exp(-x))


def der_sigm(x):
    return x * (1 - x)


def H_Calc(x_list, Wki, Ti, ne, model):
    S = 0
    for i in range(20):
        S += Wki[ne][i] * x_list[model][i]
    return sigm(S - Ti[ne])


def O_Calc(x_list, Wki, Wij, Ti, Tj, model):
    S = []
    for i in range(8):
        S.append(H_Calc(x_list, Wki, Ti, 0, model) * Wij[0][i] + H_Calc(x_list, Wki, Ti, 1, model) * Wij[1][i] - Tj[i])
    return S


def Errs_calc(e_list, res):
    Errs = [[], []]
    Errsum = 0
    for i in range(8):
        Errs[0].append(res[i] - e_list[i])
    for i in range(2):
        for k in range(8):
            Errsum += Errs[0][k] * Wij[i][k]
        Errs[1].append(Errsum)
        Errsum = 0
    return Errs


for Age in range(10000):
    for model in range(3):
        res = O_Calc(x_list, Wki, Wij, Ti, Tj, model)
        h_res = [H_Calc(x_list, Wki, Ti, 0, model), H_Calc(x_list, Wki, Ti, 1, model)]
        Errs = Errs_calc(e_list[model], res)

        for i in range(2):
            for k in range(20):
                Wki[i][k] = Wki[i][k] - ls * Errs[1][i] * der_sigm(h_res[i]) * x_list[model][k]
        for i in range(2):
            Ti[i] = Ti[i] + ls * Errs[1][i] * der_sigm(h_res[i])

        for i in range(2):
            for k in range(8):
                Wij[i][k] = Wij[i][k] - ls * Errs[0][k] * h_res[i]
        for i in range(8):
            Tj[i] = Tj[i] + ls * Errs[0][i] * res[i]

        for i in range(8):
            MSE += Errs[0][i] ** 2
    MSE = 0
inp_vector = [[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]]
print("Input:", inp_vector)
result = O_Calc(inp_vector, Wki, Wij, Ti, Tj, 0)
for i in range(8):
    print(i, ":", round(result[i], 2)*100, "%")