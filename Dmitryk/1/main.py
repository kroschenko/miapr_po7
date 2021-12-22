import math
import random
import matplotlib.pyplot as Graph

def ReferenceFunction(x):
    a, b, d = 4, 8, 0.4
    return a * math.sin(b * x) + d

def Calculate_W(w_Array, alpha, reference_Array, value, i, amount_Inputs) -> list:
    for j in range(len(w_Array)):
        w_Array[j] -= alpha * reference_Array[i + j] * (value - reference_Array[i + amount_Inputs])
    return w_Array

def Calculate_T(T, alpha, reference_Array, value, i, amount_Inputs) -> float:
    T += alpha * (value - reference_Array[i + amount_Inputs])
    return T

def GetValue(w_Array, reference_Array, T, i, amount_Inputs) -> float:
    value = 0
    for j in range(amount_Inputs):
        value += w_Array[j] * reference_Array[j + i]
    return value - T

def main():
    amount_Inputs = 3 # ---Входные нейроны
    alpha = 0.1 # =========Скорость обучения
    min_Error = 1.0e-29 # -Мин. ошибка
    step = 0.1 # ==========Шаг табуляции
    amount_Train = 30 # ---Итерации обучения
    amount_Test = 15 # ====Итерации прогнозирования
    w_Array = [ random.uniform(0, 1) for i in range(amount_Inputs) ]  # генерация весов
    T = random.uniform(0, 1)  # промежуток
    graph_Arr = []  # массив для построения графика (будут записаны результаты)
    error = 10

    # Значения эталонной функции для обучения
    reference_Array = [
        ReferenceFunction(i * step) for i in range(amount_Train)]  
    
    # Значения эталонной ф-ции для прогнозирования
    test_RefValues = [
        ReferenceFunction(i * step) for i in range(amount_Train, amount_Test + amount_Train)] 

    while error > min_Error:
        error = 0
        for i in range(amount_Train - amount_Inputs):
            value = GetValue(w_Array, reference_Array, T, i, amount_Inputs)
            w_Array = Calculate_W(w_Array, alpha, reference_Array, value, i, amount_Inputs)
            T = Calculate_T(T, alpha, reference_Array, value, i, amount_Inputs)
            error += (value - reference_Array[i + amount_Inputs]) ** 2
        error /= amount_Train - amount_Inputs
        graph_Arr.append(error)
        
    print("{:^25}{:^25}{:^25}".format("Reference value", "Predicted value", "Deviation"))
    for i in range(amount_Train - amount_Inputs):
        value = GetValue(w_Array, reference_Array, T, i, amount_Inputs)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                reference_Array[i + amount_Inputs],
                value,
                reference_Array[i + amount_Inputs] - value,
            )
        )
    print("Testing result")
    print("{:^25}{:^25}{:^25}".format("Reference value", "Predicted value", "Deviation"))
    for i in range(amount_Test - amount_Inputs):
        value = GetValue(w_Array, test_RefValues, T, i, amount_Inputs)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                test_RefValues[i + amount_Inputs],
                value,
                test_RefValues[i + amount_Inputs] - value,
            )
        )

    Graph.plot(graph_Arr)
    Graph.xlabel("Generation")
    Graph.ylabel("Error")
    Graph.grid()
    Graph.show()

main()