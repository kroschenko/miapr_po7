import numpy as np

from uninn import *


def noise(arr: np.ndarray) -> np.ndarray:
    for i in range(len(arr)):
        if np.random.uniform(0, 1) > 0.5:
            j = np.random.randint(20)
            arr[i][j] = 1 - arr[i][j]
    return arr


def main():
    l1 = LayerSigmoid(lens=(20, 10))
    l2 = LayerLinear(lens=(10, 8))
    nn = NeuralNetwork(l1, l2)
    # nn = NeuralNetwork.load('l5.nn')

    learn_x = np.array([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]])
    learn_e = np.eye(8)

    for tt in range(100):
        for t in range(999):
            nn.learn(learn_x, learn_e, alpha=0.0001, view=False)
        square_error = nn.learn(noise(learn_x), learn_e, alpha=0.0001, view=False)
        print(f'{tt + 1}000 error: {square_error.sum()}')

    for i in range(8):
        result = nn.go(learn_x[i])

        print(result.argmax())

    # nn.save('l5.nn')


if __name__ == '__main__':
    while True:
        main()
        ans = input('Еще? (y/n): ')
        if ans[0] != 'y':
            break
