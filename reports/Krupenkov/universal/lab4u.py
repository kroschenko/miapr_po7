import uninn
import numpy as np


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def main():
    # l1 = uninn.Layer(lens=(10, 4),
    #                  f_act=uninn.funsact.sigmoid,
    #                  d_f_act=uninn.funsact.d_sigmoid)
    # l2 = uninn.Layer(lens=(4, 1))
    # nn = uninn.NeuralNetwork(l1, l2)
    nn = uninn.NeuralNetwork.load('l4.nn')

    learn_x, learn_e = uninn.predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = uninn.predict_set(3, 10, 15, 0.1, function=function_lab3_9)

    for t in range(100000):
        learn_x, learn_e = uninn.shuffle_set(learn_x, learn_e)
        nn.learn(learn_x, learn_e)

    nn.go_results(test_x, test_e)
    nn.save('l4.nn')


if __name__ == '__main__':
    while True:
        main()
        ans = input('Еще? (y/n): ')
        if ans[0] != 'y':
            break
