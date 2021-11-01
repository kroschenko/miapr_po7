from uninn import *
from lab3u import function_lab3_9


def main():
    l1 = LayerSigmoid(lens=(10, 4))
    l2 = LayerLinear(lens=(4, 1))
    nn = NeuralNetwork(l1, l2)
    # nn = NeuralNetwork.load('l4.nn')

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)

    for tt in range(10):
        for t in range(999):
            nn.learn(learn_x, learn_e)
        print(f'{tt + 1}000 error: {nn.learn(learn_x, learn_e)}')

    nn.prediction_results_table(test_x, test_e)
    nn.save('l4.nn')


if __name__ == '__main__':
    while True:
        main()
        ans = input('Еще? (y/n): ')
        if ans[0] != 'y':
            break
