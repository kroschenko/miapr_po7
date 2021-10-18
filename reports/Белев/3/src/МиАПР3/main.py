from math import sin, cos, exp
from random import randrange as rand
from matplotlib import pyplot as plt


def f(x):
    return 0.1 * cos(0.1 * x) + 0.05 * sin(0.1 * x)


# Функция из условия задания


def outcalc(x, w, T):
    return w[1][0] * x[0] + w[1][1] * x[1] - T[1]


# Функция подсчета результата на выходном слое


def hiddcalc(x, w, T, model, i):
    xw_summ = 0
    for ki in range(6):
        xw_summ += x[ki + model] * w[0][ki]
    s = xw_summ - T[0][i]
    return 1 / (1 + exp(-1 * s))


# Функция подсчета результата на скрытом слое


x_list = [f(i) for i in range(36)]
e_list = [f(i) for i in range(6, 37)]
# Создаются списки образов


ls = 0.01
# for model in range(30):
#     ls_calc = 0
#     for i in range(3):
#         ls_calc += x_list[model + i] ** 2
#     ls.append(1 / (1 + ls_calc))
# # Подсчет адаптивного шага обучения


hidden_res = [0, 0]  # Результаты подсчётов на скрытом слое

Ej = 0  # Ошибка на выходном слое

Ei = [0, 0]  # Ошибка на скрытом слое

Err_list = []

w_list = [
    [rand(-10, 10) / 100 for i in range(12)],
    [rand(-10, 10) / 100 for i in range(2)]
]
# Веса инициализируются случайным образом


T_list = [
    [rand(-10, 10) / 100 for i in range(2)],
    rand(-10, 10) / 100
]
# Пороги инициализируются случайным образом


# while True:
for age in range(10000):

    MSE = 0  # Средняя квадратичная ошибка вначале эпохи равна 0

    for model in range(30):

        for i in range(2):
            hidden_res[i] = hiddcalc(x_list, w_list, T_list, model, i)
        # Подсчет результатов на скрытом слое

        y_pred = outcalc(hidden_res, w_list, T_list)
        # Подсчет результата на выходном слое

        MSE += (y_pred - e_list[model]) ** 2
        # Подсчёт суммы ошибок образов

        Ej = y_pred - e_list[model]
        # Подсчет ошибки на выходном слое

        for i in range(2):
            Ei[i] = Ej * w_list[1][i] * hidden_res[i] * (1 - hidden_res[i])
        # Подсчет ошибки на скрытом слое

        for i in range(2):
            for k in range(6):
                w_list[0][k + (6 * i)] = \
                    w_list[0][k + (6 * i)] - ls * Ei[i] * hidden_res[i] * (1 - hidden_res[i]) * x_list[k]
        # Изменение весов на i-том слое

        for j in range(2):
            w_list[1][j] = w_list[1][j] - ls * Ej * hidden_res[j]
        # Изменение весов на j-том слое

        for i in range(2):
            T_list[0][i] = T_list[0][i] + ls * Ei[i] * hidden_res[i] * (1 - hidden_res[i])
        # Изменение порогов на i-том слое

        T_list[1] = T_list[1] + ls * Ej
        # Изменение порога на j-том слое

    Err_list.append(MSE/2)

    print(MSE)
    # if MSE < 10 ** (-20):
    #     break

for i in range(2):
    hidden_res[i] = hiddcalc(x_list, w_list, T_list, 0, i)

y_pred = outcalc(hidden_res, w_list, T_list)

print(y_pred, " = ", e_list[0])

plt.plot(Err_list)
plt.show()
