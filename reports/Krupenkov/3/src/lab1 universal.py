from NeuralNetwork import *


# Функция по условию (лаб.1 вар.9)
def function_lab1_9(x: float) -> float:
    return np.sin(8 * x) + 0.3


l1 = Layer(lens=(5, 1))
nn = NeuralNetwork([l1])

learn_x, learn_e = predict_set(0, 5, 30, 0.1, function=function_lab1_9)

for t in range(20):
    for i in range(30):
        nn.learn(learn_x[i], learn_e[i], alpha=0.08)

test_x, test_e = predict_set(3, 5, 15, 0.1, function=function_lab1_9)
nn.go_results(test_x, test_e)
