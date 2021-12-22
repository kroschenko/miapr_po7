from math import *
from random import random as r


def formula(x):
    return 0.1 * cos(0.1 * x) + 0.05 * sin(0.1 * x)


def sigm(x):
    return 1 / (1 + exp(-x))


def sigm_der(x):
    return x * (1 - x)


x_list = [formula(i) for i in range(55)]
e_list = [formula(i) for i in range(6, 56)]
test_list = [formula(i) for i in range(56, 73)]
Wki = [[r() for i in range(6)], [r() for i in range(6)]]
Wij = [r() for i in range(2)]
T = [r() for i in range(3)]
h_Summ = [0, 0]
ls = 0.1
Errs = [0, 0]
MSE = 0
x_summ = 0
Age = 0

def hcalc(Wki, T, model, i, x_list):
    res = 0
    for ne in range(6):
        res += Wki[i][ne] * x_list[ne + model]
    return sigm(res - T[i])


def ocalc(Wij, T, h_Summ):
    return Wij[0] * h_Summ[0] + Wij[1] * h_Summ[1] - T[2]


while True:
    Age += 1
    for model in range(50):



        for i in range(2):
            h_Summ[i] = hcalc(Wki, T, model, i, x_list)
        strcalc = ocalc(Wij, T, h_Summ)
        Err = strcalc - e_list[model]
        MSE += Err ** 2

        for i in range(2):
            Errs[i] = Err * Wij[i]

        for i in range(6):
            x_summ += x_list[i + model] ** 2
        temp1 = ((Errs[0] ** 2) * sigm_der(h_Summ[0])) + ((Errs[1] ** 2) * sigm_der(h_Summ[1]))
        temp2 = (Errs[0] * sigm_der(h_Summ[0]))**2 + (Errs[1] * sigm_der(h_Summ[1]))**2
        hls = (4 * temp1) / ((1 + x_summ) * temp2)

        ls = ((Err**2) / (1 + Err ** 2))

        for i in range(2):
            Wij[i] = Wij[i] - ls * Err * h_Summ[i]
        T[2] = T[2] + ls * Err

        for i in range(2):
            for k in range(6):
                Wki[i][k] = Wki[i][k] - hls * Errs[i] * sigm_der(h_Summ[i]) * x_list[k+model]
        for i in range(2):
            T[i] = T[i] + hls * Errs[i] * sigm_der(h_Summ[i])
    if MSE < 10e-5:
        break;
    MSE = 0
    x_summ = 0
for test_model in range(10):
    for i in range(2):
        h_Summ[i] = hcalc(Wki, T, test_model, i, test_list)
    strcalc = ocalc(Wij, T, h_Summ)
    print(strcalc, " = ", test_list[6 + test_model])
print("Эпох прошло:", Age)
