from math import sin, cos, exp
import random
import matplotlib.pyplot as plt

MIN_ERROR = 1e-6
SPEED_TRAINING = 0.09

input_neuron_number = 8
hidden_neuron_number = 3

Wij = [[random.uniform(-1, 1) for _ in range(hidden_neuron_number)] for _ in range(input_neuron_number - 1)]
Wjk = [random.uniform(-1, 1) for _ in range(hidden_neuron_number)]

Tj = [0 for _ in range(hidden_neuron_number)]
Tk = 0


def function(x):
    a, b, c, d = 0.4, 0.2, 0.07, 0.2
    return a * cos(b * x * 0.1) + c * sin(d * x * 0.1)


def input_elements(x):
    y = []
    etalon_y = []
    for i in range(input_neuron_number):
        if i == input_neuron_number - 1:
            y.append(etalon_y)
            y.append(function(x))
        else:
            etalon_y.append(function(x))
            x += 1
    return y


def Sj_onHiddenLayer(j, y):
    Sj = 0
    for i in range(input_neuron_number - 1):
        Sj += y[0][i] * Wij[i][j]
    Sj -= Tj[j]
    return Sj


def sigmoid_function(j, y):
    Sj = Sj_onHiddenLayer(j, y)
    return 1 / (1 + exp(-Sj))


def Sk_onOutputLayer(Yj):
    Sk = 0
    for i in range(hidden_neuron_number):
        Sk += Yj[i] * Wjk[i]
    Sk -= Tk
    return Sk


def Wjk_change(Yj, error):
    for i in range(hidden_neuron_number):
        Wjk[i] -= SPEED_TRAINING * error * Yj[i]


def Wij_change(Yj, error, y):
    for i in range(input_neuron_number - 1):
        for j in range(hidden_neuron_number):
            Wij[i][j] -= SPEED_TRAINING * error * Wjk[j] * y[0][i] * (Yj[j] * (1 - Yj[j]))


def Tj_change(Yj, error):
    for i in range(hidden_neuron_number):
        Tj[i] += SPEED_TRAINING * error * Yj[i] * (1 - Yj[i])


def main():
    epoch = 0
    sum_error = 1
    global Tk
    drawing_data = ([], [])

    print("Результаты обучения:")
    print("{:<30}{:<30}{:<30}{}".format("Эталонное значение", "Текущее значение", "Ошибка", "Суммарная ошибка"))

    while sum_error > MIN_ERROR:
        for k in range(30):
            Yj, y = [], []
            Yk, sum_error = 0, 0
            y = input_elements(k)
            for i in range(hidden_neuron_number):
                Yj.append(sigmoid_function(i, y))
            Yk = Sk_onOutputLayer(Yj)
            local_error = Yk - y[-1]
            Wjk_change(Yj, local_error)
            Wij_change(Yj, local_error, y)
            Tj_change(Yj, local_error)
            Tk += local_error * SPEED_TRAINING
            sum_error += 0.5 * (local_error ** 2)

        epoch += 1
        if epoch % 250 == 0:
            print(f"{y[-1]:<29} {Yk:<29} {local_error:<29} {sum_error}")

        drawing_data[0].append(epoch)
        drawing_data[1].append(sum_error)
    print(f"Количество эпох: {epoch}")

    print(f"\nWjk = {Wjk}")
    print(f"\nWij = {Wij}")
    print(f"\nTj = {Tj}")
    print(f"\nTk = {Tk}")

    print("\n\nРезультаты тестирования:")
    print("{:<30}{:<30}{}".format("Эталонное значение", "Текущее значение", "Погрешность"))
    for i in range(30, 45):
        Yj_, y_ = [], []
        Yk_, sum_error_ = 0, 0
        y_ = input_elements(i)
        for j in range(hidden_neuron_number):
            Yj_.append(sigmoid_function(j, y_))
        Yk_ = Sk_onOutputLayer(Yj_)
        local_error_ = Yk_ - y_[-1]
        sum_error_ += 0.5 * (local_error_ ** 2)

        print(f"{y_[-1]:<29} {Yk_:<29} {local_error_}")

    plt.plot(*drawing_data)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    main()
