from uninn import *


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def repeat_func(func):
    while True:
        func()
        ans = input('Еще? (y/n): ')
        if not ans or (ans[0] != 'y' and ans != 'н'):
            break


def main():
    l1 = LayerSigmoid(lens=(10, 4))
    l2 = LayerLinear(lens=(4, 1))
    nn = NeuralNetwork(l1, l2)
    # nn = NeuralNetwork.load('l3.nn')

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    alpha = 0.2

    import time
    start_time = time.time()

    for thousand in range(20):
        for _ in range(999):
            nn.learn(learn_x, learn_e, alpha)
        print(f'{thousand + 1},000 error: {nn.learn(learn_x, learn_e, alpha)}')

    print("--- %s seconds ---" % (time.time() - start_time))

    nn.prediction_results_table(test_x, test_e)
    nn.save('l3.nn')


if __name__ == '__main__':
    repeat_func(main)
