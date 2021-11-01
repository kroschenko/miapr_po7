from uninn import *
from lab1u import function_lab1_9


def main():
    l1 = LayerLinear(lens=(5, 1))
    nn = NeuralNetwork(l1)

    learn_x, learn_e = predict_set(0, 5, 30, 0.1, function=function_lab1_9)
    for t in range(10):
        square_error = nn.learn(learn_x, learn_e)
        print(f'Square error {t}: {square_error}')

    test_x, test_e = predict_set(3, 5, 15, 0.1, function=function_lab1_9)
    nn.prediction_results_table(test_x, test_e)


if __name__ == '__main__':
    main()
