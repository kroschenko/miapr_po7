from math import *
from random import random as rn


def func(x):
    a = 0.2
    b = 0.2
    c = 0.06
    d = 0.2
    return (a*cos(b*x) + c*sin(d*x))

def proizv(y):
    return y*(1 - y)

def sigm(S):
    return(1 + exp(-S))

inputs = 8
ed = 0.01
E_min = 0.001
W_ki = [rn() for i in range(24)]
W_ij = [rn() for i in range(3)]
T = [rn() for i in range(4)]
train = [func(i) for i in range(58)]
test = [func(i) for i in range(58, 81)]
E = 1

while E > E_min:
    E = 0
    for i in range(38 - inputs):
       
        y_i = [0, 0, 0]
        
        # Прямое распространение для скрытого слоя
        for j in range(3):
            S_i = 0
            for k in range(inputs):
                S_i += W_ki[k + inputs*j] * train[i+k]
            S_i -= T[j]
            y_i[j] = sigm(S_i)
       
        y_j = 0

        # Прямое распространение для выходного слоя
        for k in range(len(y_i)):
            y_j += W_ij[k] * y_i[k]
        y_j -= T[3]

        # Обратное распространение ошибки
        Err_j = y_j - train[inputs + i]
        Err_i = [0, 0, 0]
        for n in range(len(Err_i)):
            Err_i[n] = Err_j * W_ij[n]

        # Корректировка весов и порогов
        for k in range(len(Err_i)):
            for n in range(inputs):
                W_ki[n+inputs*k] -= ed*Err_i[k]*proizv(y_i[k])*train[i+n]
            T[k] += ed*Err_i[k]*proizv(y_i[k])

        for k in range(len(y_i)):
            W_ij[k] -= ed*Err_j*y_i[k]
        T[3] += ed*Err_j

        # Вычисление среднеквадрaтичной ошибки
        E += pow(y_j - train[inputs+i], 2)
    E/=2

for i in range(58, 81 - inputs):
       
        y_i = [0, 0, 0]
       
        for j in range(3):
            S_i = 0
            for k in range(inputs):
                S_i += W_ki[k + inputs*j] * test[i+k-58]
            S_i -= T[j]
            y_i[j] = sigm(S_i)
       
        y_j = 0

        for k in range(len(y_i)):
            y_j += W_ij[k] * y_i[k]
        y_j -= T[3]

        print(y_j, " = ", test[inputs+i-58])