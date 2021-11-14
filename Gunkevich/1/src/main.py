import random
import math
import matplotlib.pyplot as plt

def function(x):
    return 3 * math.sin(7*x) + 0.3

def neiro_func(inputs, weights, t, x, shift):
    output = 0
    for i in range(inputs):
        output += weights[i]*x[i+shift]
    return output - t


inputs = 5
training = 35
testing = 20
step = 0.1
training_outputs = [function(i*step) for i in range(training)]
testing_outputs = [function(i*step) for i in range(training, testing+training)]
weights = [random.uniform(-1, 1) for i in range(inputs)]
t = random.uniform(0, 1)
age = 0
min_error = 0.00009
ed_speed = 0.01
error = 1
graf = []
graf2 = []
while error > min_error:
    error = 0
    age += 1
    for i in range(training-inputs):
        output = neiro_func(inputs, weights, t, training_outputs, i)
        expected_output = training_outputs[inputs+i]
        for j in range(inputs):
            weights[j] -= ed_speed * (output - expected_output) * training_outputs[j + i]
        t += ed_speed * (output - expected_output)
        error += (output - expected_output) ** 2
        error /= 2
        graf.append(error)
        graf2.append(i)
print(f"Итоговые веса: {weights}\nИтоговый порог Т = {t}\nЧисло эпох = {age}")
print("Результаты прогнозирования:")
print("Эталонное значение\t\t Полученные значения\t\t Отклонение\t")
for i in range(testing - inputs):
    output = neiro_func(inputs, weights, t, testing_outputs, i)
    print(f"{output:25} {testing_outputs[i+inputs]:25} {testing_outputs[i+inputs] - output:25}")
plt.plot(graf)
plt.show()