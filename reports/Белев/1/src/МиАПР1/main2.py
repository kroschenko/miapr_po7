from math import sin


def f(x):
    return sin(8 * x) + 0.3

def output(x,w, b, form):
    yo = 0
    for i in range(5):
        yo += w[i] * x[i + form]
    return yo+b


n_model = 35
list_x = [x / 10 for x in range(n_model)]
list_y = [f(x) for x in range(n_model)]
# print(list_x, list_y, sep='\n')
learning_speed = 0.1
list_w = [0.1, -0.1, 0.1, 0.1, 0]
T = 1
E = 1
E_wanted = 0.1
for k in range(2):
    print(E)
    for form in range(30):
        y = output(list_y, list_w, T, form)
        E += (y - list_y[form + 5]) ** 2
        for i in range(5):
            list_w[i] = list_w[i] - learning_speed * (y - list_y[form + 5]) * list_y[i + form]
        T = T + learning_speed * (y - list_y[form + 5])
        y = output(list_y, list_w, T, form)
print(E, " ", T, " ", list_w)
print(f(list_y[34]), " ", output(list_y, list_w, T, 29))
