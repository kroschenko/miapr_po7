import random
from math import sin, cos, exp

def Reference_f(x):
    a, b, c, d = 0.2, 0.6, 0.05, 0.6
    return a * cos(b * x) + c * sin(d * x)

def Calculate_S_Hidden(arr_Hidden_Y, arr_Hidden_W, arr_RefValues, arr_Hidden_T, amount_Input, amount_Hidden, iter):
    for i in range(len(arr_Hidden_Y)):
        arr_Hidden_Y[i] = 0

    for i in range(amount_Hidden):
        for j in range(amount_Input):
            arr_Hidden_Y[i] += arr_Hidden_W[i][j] * arr_RefValues[iter + j]
        arr_Hidden_Y[i] -= arr_Hidden_T[i]
    return arr_Hidden_Y

def Calculate_Y_Hidden(S_hid):
    return 1 / 1 + exp(-S_hid)

def Calculate_y(arr_Hidden_Y, arr_W_Out, t_Out, amount_Hidden):
    y = 0
    for i in range(amount_Hidden):
        y += arr_Hidden_Y[i] * arr_W_Out[i]
    return y - t_Out

def change_w_out(arr_W_Out, gamma_Out, arr_Hidden_Y, amount_Hidden):
    alpha = 0.1
    for i in range(amount_Hidden):
        arr_W_Out[i] -= alpha * gamma_Out * arr_Hidden_Y[i]
    return arr_W_Out

def change_t_Out(t_Out, gamma_Out):
    alpha = 0.1
    t_Out += alpha * gamma_Out
    return t_Out

def change_w_Hidden(
    arr_Hidden_W, gamma_Hidden, arr_Hidden_Y, arr_RefValues, amount_Input, amount_Hidden, iter
):
    alpha = 0.1
    for i in range(amount_Hidden):
        for j in range(amount_Input):
            arr_Hidden_W[i][j] -= (
                alpha
                * gamma_Hidden[i]
                * arr_Hidden_Y[i]
                * (1 - arr_Hidden_Y[i])
                * arr_RefValues[j + iter]
            )
    return arr_Hidden_W

def change_T_Hidden(arr_Hidden_T, gamma_Hidden, arr_Hidden_Y, amount_Hidden):
    alpha = 0.1
    for i in range(amount_Hidden):
        arr_Hidden_T[i] += alpha * gamma_Hidden[i] * arr_Hidden_Y[i] * (1 - arr_Hidden_Y[i])
    return arr_Hidden_T

def main():
    step = 0.1
    amount_Input = 10
    amount_Hidden = 4
    amount_Train = 30
    amount_Test = 15
    
    arr_RefValues = [Reference_f(x * step) for x in range(amount_Train)]
    arr_Hidden_W = []
    for i in range(amount_Hidden):
        row = []
        for j in range(amount_Input):
            row.append(random.uniform(0, 1))
        arr_Hidden_W.append(row)
    print('Hidden weights: ',arr_Hidden_W)

    arr_Hidden_T = [0 for i in range(amount_Hidden)]
    arr_Hidden_Y = [0 for i in range(amount_Hidden)]

    arr_W_Out = [random.uniform(0, 1) for i in range(amount_Hidden)]
    t_Out = random.uniform(0, 1)
    
    gamma_Out = 0
    gamma_Hidden = [0 for i in range(amount_Hidden)]
    
    gen_Count = 0
    y = 0
    error = 1
    min_error = 1e-6
    while error > min_error:
        error = 0
        for iter in range(amount_Train - amount_Input):

            arr_Hidden_Y = Calculate_S_Hidden(
                arr_Hidden_Y, arr_Hidden_W, arr_RefValues, arr_Hidden_T, amount_Input, amount_Hidden, iter
            )
            for i in range(amount_Hidden):
                arr_Hidden_Y[i] = Calculate_Y_Hidden(arr_Hidden_Y[i])

            y = Calculate_y(arr_Hidden_Y, arr_W_Out, t_Out, amount_Hidden)

            gamma_Out = y - arr_RefValues[iter + amount_Input]

            for i in range(amount_Hidden):
                gamma_Hidden[i] = gamma_Out * y * (1 - y) * arr_W_Out[i]

            arr_W_Out = change_w_out(arr_W_Out, gamma_Out, arr_Hidden_Y, amount_Hidden)
            t_Out = change_t_Out(t_Out, gamma_Out)
            arr_Hidden_W = change_w_Hidden(
                arr_Hidden_W, gamma_Hidden, arr_Hidden_Y, arr_RefValues, amount_Input, amount_Hidden, iter
            )
            arr_Hidden_T = change_T_Hidden(arr_Hidden_T, gamma_Hidden, arr_Hidden_Y, amount_Hidden)

            error += ((y - arr_RefValues[iter + amount_Input]) ** 2) / 2
        gen_Count += 1
        error /= amount_Train - amount_Input
        if gen_Count % 5000 == 0:
            print('Error value: ', error)

    print("Training end\nTraining result")
    print("{:^25}{:^25}{:^25}".format("Reference value", "Prediction value", "Deviation"))
    for iter in range(amount_Train - amount_Input):
        arr_Hidden_Y = Calculate_S_Hidden(arr_Hidden_Y, arr_Hidden_W, arr_RefValues, arr_Hidden_T, amount_Input, amount_Hidden, iter)
        for i in range(amount_Hidden):
            arr_Hidden_Y[i] = Calculate_Y_Hidden(arr_Hidden_Y[i])
        y = Calculate_y(arr_Hidden_Y, arr_W_Out, t_Out, amount_Hidden)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                arr_RefValues[iter + amount_Input],
                y,
                arr_RefValues[iter + amount_Input] - y,
            )
        )
    ref_test_arr = [
        Reference_f(x * step) for x in range(amount_Train - amount_Input, amount_Test + amount_Train)
    ]
    print("Testing result")
    print("{:^25}{:^25}{:^25}".format("Reference value", "Prediction value", "Deviation"))
    for iter in range(amount_Test):
        arr_Hidden_Y = Calculate_S_Hidden(
            arr_Hidden_Y, arr_Hidden_W, ref_test_arr, arr_Hidden_T, amount_Input, amount_Hidden, iter
        )
        for i in range(amount_Hidden):
            arr_Hidden_Y[i] = Calculate_Y_Hidden(arr_Hidden_Y[i])
        y = Calculate_y(arr_Hidden_Y, arr_W_Out, t_Out, amount_Hidden)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                ref_test_arr[iter + amount_Input],
                y,
                ref_test_arr[iter + amount_Input] - y,
            )
        )


main()
