from math import *
from random import random as rn

def func(x):
    a = 2
    b = 6
    d = 0.2
    return a*sin(b*x) + d


inputs = 4
train_sample = [func(i+0.1) for i in range(34)]
test_sample = [func(i+0.1) for i in range(34, 53)]
speed_of_edu = 0.01
E_min = 0.000001
W = [rn() for i in range(4)]
T = rn()
print(W)
print(T)
E = 1
num_of_epochs = 0

while E > E_min:
    E = 0
    num_of_epochs+=1
    for i in range(34 - inputs):
        output = 0
        for j in range(inputs):
            output += W[j]*train_sample[j+i]
        output -= T
        E += 0.5 * pow((output - train_sample[inputs + i]), 2)
        ada_step = 1
        # Корректировка параметров
        for n in range(inputs):
            ada_step += pow(train_sample[n+i], 2)
        ada_step = 1 / ada_step
        if E > E_min:
            for k in range(inputs):
                W[k] = W[k] - ada_step*(output - train_sample[inputs + i]) * train_sample[k+i]
            T = T + ada_step * (output - train_sample[inputs + i])

print("Кол-во эпох = ", num_of_epochs)

for i in range(34, 53 - inputs):
    test_out = 0
    for j in range(inputs):
        test_out+=W[j]*test_sample[i-34+j]
    test_out -= T
    print(test_out, " = ", test_sample[inputs+i-34])