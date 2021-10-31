from uninn import *


# Функция по условию (лаб.1 вар.9)
def function_lab1_9(x: float) -> float:
    return np.sin(8 * x) + 0.3


def main():
    l1 = LayerLinear(lens=(5, 1))
    nn = NeuralNetwork(l1)

    learn_x, learn_e = predict_set(0, 5, 30, 0.1, function=function_lab1_9)
    for t in range(10):
        square_error = nn.learn(learn_x, learn_e)
        print(f'Square error {t}: {square_error}')

    test_x, test_e = predict_set(3, 5, 15, 0.1, function=function_lab1_9)
    nn.go_results(test_x, test_e)


if __name__ == '__main__':
    main()