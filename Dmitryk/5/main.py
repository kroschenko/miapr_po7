import random
import math
import matplotlib.pyplot as Graph

def sigm_Function(S):
    return 1 / (1 + math.exp(-S))

def Sj_Hidden(y):
    Sj = []
    for j in range(hidden_neuron_number):
        value = 0
        for i in range(input_neuron_number - 1):
            value += y[i] * Wij[i][j]
        value -= Tj[j]
        Sj.append(sigm_Function(value))
    return Sj

def Sk_Output(Yj):
    Sk = []
    for j in range(output_neuron_number):
        value = 0
        for i in range(hidden_neuron_number):
            value += Yj[i] * Wjk[j][i]
        value -= Tk[j]
        Sk.append(sigm_Function(value))
    return Sk

def Wjk_Change(Yj, Yk, error):
    global Tk
    for j in range(output_neuron_number):
        for i in range(hidden_neuron_number):
            Wjk[j][i] -= step * error[j] * Yk[j] * (1 - Yk[j]) * Yj[i]
        Tk[j] += error[j] * step * Yk[j] * (1 - Yk[j])
        
def Wij_Change(Yj, Hidden_error, y):
    for j in range(hidden_neuron_number):
        for i in range(input_neuron_number - 1):
            Wij[i][j] -= step * Hidden_error[j] * y[i] * Yj[j] * (1 - Yj[j])
        Tj[j] += step * Hidden_error[j] * Yj[j] * (1 - Yj[j])

vector_4 = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]
vector_3 = [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
vector_8 = [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1]
all_vectors = [vector_4, vector_3, vector_8]
step = 0.2
min_error = 1e-5
input_neuron_number = 10
hidden_neuron_number = 4
output_neuron_number = 1
Wij = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(input_neuron_number - 1)]
Wjk = [[random.uniform(-0.1, 0.1) for _ in range(hidden_neuron_number)] for _ in range(output_neuron_number)]
Tj = [random.uniform(-0.5, 0.5) for _ in range(hidden_neuron_number)]
Tk = [random.uniform(-0.5, 0.5) for _ in range(output_neuron_number)]

def main():
    
    arr_Graph = ([], [])
    errors = [0] * output_neuron_number
    reference = [0] * output_neuron_number
    Hidden_error = [0] * hidden_neuron_number
    iteration = 1
    generation = 0
    error = 1
    while error > min_error:
        error = 0
        for N in range(output_neuron_number):
            reference[N] = 1
            for i in range(iteration):
                y = all_vectors[N]
                Yj = Sj_Hidden(y)
                Yk = Sk_Output(Yj)
                for index in range(output_neuron_number):
                    errors[index] = Yk[index] - reference[index]
                for j in range(hidden_neuron_number):
                    for k in range(output_neuron_number):
                        Hidden_error[j] += errors[k] * Yk[k] * (1 - Yk[k]) * Wjk[k][j]
                Wjk_Change(Yj, Yk, errors)
                Wij_Change(Yj, Hidden_error, y)
                error += errors[N] ** 2
        error /= 2
        arr_Graph[0].append(generation)
        arr_Graph[1].append(error)
        generation += 1
    Graph.plot(*arr_Graph)
    Graph.xlabel("generation")
    Graph.ylabel("Error")
    Graph.show()
    for i in range(len(all_vectors)):
        input = all_vectors[i]
        print("Result vector :", i + 1, end=" : ")
        for j in range(len(vector_4)):
            print(input[j], end='')
        print("\nResult : ", end='')
        Hidden_prev = Sj_Hidden(input)
        Values = Sk_Output(Hidden_prev)
        print(Values[0])

main()
