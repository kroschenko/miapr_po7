from math import sin
from matplotlib import pyplot as plt

def f(x):
    return sin(5 * x) + 0.1
# Ф-ия расчета значения у


def Preditiction(x, w, t, model):
    return x[model] * w[0] + x[model + 1] * w[1] + x[model + 2] * w[2] - t
# Ф-ия расчета предсказания


list_model_y = []
list_model_x = []
w = [0.1, -0.1, 0]
t = 0.2
Age = 0
ls = []
list_E = []


for i in range(33):
    list_model_x.append(i / 10)
for i in range(33):
    list_model_x[i] = f(list_model_x[i])
for i in range(3, 33):
    list_model_y.append(list_model_x[i])
# создание списков с входными и эталонными значениями


for model in range(30):
    ls_calc = 0
    for i in range(3):
        ls_calc += list_model_x[model + i] ** 2
    ls.append(1 / (1 + ls_calc))
# Подсчет шагов обучения для разных входных образов


while True:
    Age += 1
    E = 0
    for model in range(30):
        y_pred = Preditiction(list_model_x, w, t, model)
        for i in range(3):
            w[i] = w[i] - ls[model] * (y_pred - list_model_y[model]) * list_model_x[model + i]
        t = t + ls[model] * (y_pred - list_model_y[model])
        E += (y_pred - list_model_y[model]) ** 2
    list_E.append(E)
    if E < 1E-10:
        break
# Обучение персептрона


print("Предсказанное\t\t\t  Эталонное")
for i in range(15):
    print(f"{Preditiction(list_model_x, w, t, i):<25} {list_model_y[i]:<25}")
print("Прошло ", Age, " Эпох")
plt.plot(list_E)
plt.show()