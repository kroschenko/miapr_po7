from uninn import *


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def main():
    l1 = Layer(lens=(10, 4),
               f_act=funsact.sigmoid,
               d_f_act=funsact.d_sigmoid)
    l2 = Layer(lens=(4, 1))
    nn = NeuralNetwork(l1, l2)
    # nn = NeuralNetwork.load('l3.nn')

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    alpha = 0.01

    for tt in range(100):
        for t in range(999):
            nn.learn(learn_x, learn_e, alpha)
        print(f'{tt + 1}000 error: {nn.learn(learn_x, learn_e, alpha)}')

    nn.prediction_results_table(test_x, test_e)
    nn.save('l3.nn')


if __name__ == '__main__':
    while True:
        main()
        ans = input('Еще? (y/n): ')
        if ans[0] != 'y':
            break
