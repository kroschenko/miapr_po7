import math
import random
import matplotlib.pyplot as Graph

def ReferenceFunction(x):
    a, b, d = 4, 8, 0.4
    return a * math.sin(b * x) + d

def Calculate_W(w_arr, alpha, reference_Array, value, i, amount_Inputs) -> list:
    for j in range(len(w_arr)):
        w_arr[j] -= alpha * reference_Array[i + j] * (value - reference_Array[i + amount_Inputs])
    return w_arr

def Calculate_T(T, alpha, reference_Array, value, i, amount_Inputs) -> float:
    T += alpha * (value - reference_Array[i + amount_Inputs])
    return T

def GetValue(w_arr, reference_Array, T, i, amount_Inputs) -> float:
    value = 0
    for j in range(amount_Inputs):
        value += w_arr[j] * reference_Array[j + i]
    return value - T

def Calculate_Alpha(etalon_arr, i, amount_inputs) -> float:
    alpha = 0
    for j in range(amount_inputs):
        alpha += etalon_arr[j + i] ** 2
    return (1 / (1 + alpha))

def main():
    amount_Inputs = 3 # ---Входные нейроны
    min_Error = 1.0e-29 # -Мин. ошибка
    step = 0.1 # ==========Шаг табуляции
    amount_Train = 30 # ---Итерации обучения
    amount_Test = 15 # ====Итерации прогнозирования
    w_arr = [ random.uniform(0, 1) for i in range(amount_Inputs) ]  # генерация весов
    T = random.uniform(0, 1)  # промежуток
    graph_Arr = []  # массив для построения графика (будут записаны результаты)
    error = 10

    # Значения эталонной функции для обучения
    reference_Array = [
        ReferenceFunction(i * step) for i in range(amount_Train)]  
    
    # Значения эталонной ф-ции для прогнозирования
    test_RefValues = [
        ReferenceFunction(i * step) for i in range(amount_Train, amount_Test + amount_Train)] 

    Gen_iteration = 0
    while error > min_Error:
        error = 0
        for i in range(amount_Train - amount_Inputs):
            value = GetValue(w_arr, reference_Array, T, i, amount_Inputs)
            alpha = Calculate_Alpha(reference_Array, i, amount_Inputs)
            w_arr = Calculate_W(w_arr, alpha, reference_Array, value, i, amount_Inputs)
            T = Calculate_T(T, alpha, reference_Array, value, i, amount_Inputs)
            error += (value - reference_Array[i + amount_Inputs]) ** 2
        error /= amount_Train - amount_Inputs
        Gen_iteration += 1
        graph_Arr.append(error)

    print("Training end\nTraining result")
    print("{:^25}{:^25}{:^25}".format("reference value", "predicted value", "Deviation"))
    for i in range(amount_Train - amount_Inputs):
        value = GetValue(w_arr, reference_Array, T, i, amount_Inputs)
        print(
            "{:< 25}{:< 25}{:< 25}".format(
                reference_Array[i + amount_Inputs],
                value,
                reference_Array[i + amount_Inputs] - value,
            )
        )
    print("Testing result:", Gen_iteration, "epoch")
    print("{:^25}{:^25}{:^25}".format("reference value", "predicted value", "Deviation"))
    for i in range(amount_Test - amount_Inputs):
        value = GetValue(w_arr, test_RefValues, T, i, amount_Inputs)
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