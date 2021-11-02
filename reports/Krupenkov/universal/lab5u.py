from uninn import *
from lab3u import repeat_func


def noise(arr: np.ndarray) -> np.ndarray:
    for i in range(len(arr)):
        if np.random.uniform(0, 1) > 0.5:
            j = np.random.randint(20)
            arr[i][j] = 1 - arr[i][j]
    return arr


def noise_j(arr: np.ndarray, j) -> np.ndarray:
    arr[j] = 1 - arr[j]
    return arr


def main():
    # relu = funsact.Relu(k=0.1)
    l1 = LayerLinear(lens=(20, 20))
    l2 = LayerLinear(lens=(20, 8))
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

    for thousand in range(100):
        for _ in range(999):
            nn.learn(noise(learn_x), learn_e)
        print(f'{thousand + 1},000 error: {nn.learn(noise(learn_x), learn_e).sum()}')

    # for i in range(8):
    #     print(f'[{i}]: {nn.go(learn_x[i]).argmax()}')

    print('Глубокая проверка:')
    correct_amount = 0
    for i in range(8):
        row = learn_x[i]
        print(f'[{i}]: {nn.go(learn_x[i]).argmax()} | ', end='')
        for j in range(20):
            out = nn.go(noise_j(row, j)).argmax()
            if out == i:
                correct_amount += 1
            print(out, end=' ')
        print()
    print(f'Правильно {correct_amount} / 160: {correct_amount / 1.6 : .1f}%')

    nn.save('l5.nn')


if __name__ == '__main__':
    repeat_func(main)
