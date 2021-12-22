from math import *
from random import random as rn

def proizv(y):
    return y*(1 - y)

def sigm(S):
    return(1 + exp(-S))

vectors = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]]

ideals = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]

W_ki = [[rn() for i in range(20)], [rn() for i in range(20)], [rn() for i in range(20)]]
W_ij = [[rn() for i in range(3)], [rn() for i in range(3)], [rn() for i in range(3)]]
T_i = [rn() for i in range(3)]
T_j = [rn() for i in range(3)]
ed = 0.01
for epoch in range(10000):
    for i in range(len(vectors)):
        y_i = [0, 0, 0]
        for j in range(3):
            S_i = 0
            for k in range(20):
                S_i += W_ki[j][k] * vectors[i][k]
            S_i -= T_i[j]
            y_i[j] = sigm(S_i)
        
        y_j = [0, 0, 0]
        for j in range(3):
            for k in range(3):
                y_j[j] += W_ij[j][k] * y_i[k]
            y_j[j] -= T_j[j]

        Err_j = [0,0,0]
        for j in range(3):
            Err_j[j] = y_j[j] - ideals[i][j]

        Err_i = [0,0,0]
        for j in range(3):
            for k in range(3):
                Err_i[j] += Err_j[j]*W_ij[j][k]
        for j in range(3):
            for k in range(20):
                W_ki[j][k] -= ed*Err_i[j] * y_i[j]*(1 - y_i[j])*vectors[i][k]
            T_i[j] += ed*Err_i[j] * y_i[j]*(1 - y_i[j])

        for j in range(3):
            for k in range(3):
                W_ij[j][k] -= ed*Err_j[j] * y_i[k]
            T_j[j] += ed*Err_j[j]

y_i = [0, 0, 0]
for j in range(3):
    S_i = 0
    for k in range(20):
       S_i += W_ki[j][k] * vectors[0][k]
    S_i -= T_i[j]
    y_i[j] = sigm(S_i)
        
y_j = [0, 0, 0]
for j in range(3):
    for k in range(3):
        y_j[j] += W_ij[j][k] * y_i[k]
    y_j[j] -= T_j[j]


print(y_i, " = ", ideals[0])