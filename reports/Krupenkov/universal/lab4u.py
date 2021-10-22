from uninn_adaptive import *


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def main():
    l1 = Layer(lens=(10, 4), f_act=sigmoid, d_f_act=d_sigmoid)
    l2 = Layer(lens=(4, 1))
    nn = NeuralNetwork([l1, l2])
    # nn = NeuralNetwork.load('l4.nn')

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)

    for t in range(1000):
        learn_x, learn_e = shuffle_set(learn_x, learn_e)
        for i in range(30):
            nn.learn(learn_x[i], learn_e[i])

    nn.go_results(test_x, test_e)
    nn.save('l4.nn')


if __name__ == '__main__':
    main()
