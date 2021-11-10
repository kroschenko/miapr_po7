from uninn import *
import time
from lab3u import repeat_func


def noise(arr: np.ndarray) -> np.ndarray:
    for i in range(len(arr)):
        if np.random.randint(20) < 1:
            j = np.random.randint(20)
            arr[i][j] = 1 - arr[i][j]
    return arr


def noise_j(arr: np.ndarray, j) -> np.ndarray:
    arr[j] = 1 - arr[j]
    return arr


def main():
    l1 = LayerSigmoid(lens=(20, 50))
    l2 = LayerLinear(lens=(50, 8))
    nn = NeuralNetwork(l1, l2)

    learn_x = np.array(
        [
            [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        ]
    )
    learn_e = np.eye(8)
    times = 20_000
    sep = 1_000
    print(f"- Learning {times} times -")

    start_time = time.time()

    for thousand in range(times // sep):
        for _ in range(sep - 1):
            nn.learn(noise(learn_x), learn_e)
        print(
            f"{thousand + 1 : 3},000 error: {nn.learn(noise(learn_x), learn_e).sum()}"
        )

    print(f"- Learning time: {time.time() - start_time} seconds -")

    print(
        "Глубокая проверка с заменой каждого бита\n"
        "[i]:orgnl|  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19"
    )
    correct_amount = 0
    for i in range(8):
        row = learn_x[i]
        out = (nn.go(row)).argmax()
        print(f"[{i}]:  {out}  |  ", end="")
        if out == i:
            correct_amount += 1

        for j in range(20):
            out = (nn.go(noise_j(row, j))).argmax()
            if out == i:
                correct_amount += 1
            print(out, end="  ")
        print()
    print(f"Правильно {correct_amount} / 168: {correct_amount / 1.68 : .1f}%")


if __name__ == "__main__":
    # repeat_func(main)
    main()
