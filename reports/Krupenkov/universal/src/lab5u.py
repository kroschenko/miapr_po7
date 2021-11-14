from uninn import *
import time


def switch_bit(row: np.ndarray, j) -> np.ndarray:
    row[j] ^= 1
    return row


def main():
    nn = NeuralNetwork(
        LayerSigmoid(lens=(20, 4)),
        LayerLinear(lens=(4, 3))
    )

    learn_x = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        ]
    )
    learn_e = np.eye(3)
    times = 2
    print(f"- Learning {times} times -")

    start_time = time.time()

    for t in range(times):
        nn.learn(learn_x, learn_e)
        error = nn.learn(learn_x, learn_e).sum()
        print(f"{t + 1 : 5d}/{times} error: {error : .5e}")
    print(f"- Learning time: {time.time() - start_time} seconds -")

    print("\nГлубокая проверка с заменой битов №\n[i]: - | 0 1 2 3 ...")
    correct_amount = 0
    for i in range(3):
        row = learn_x[i]
        out = (nn.go(row)).argmax()
        print(f"[{i}]: {out} | ", end="")
        if out == i:
            correct_amount += 1

        for j in range(20):
            out = (nn.go(switch_bit(row, j))).argmax()
            if out == i:
                correct_amount += 1
            print(out, end=" ")
        print()
    print(f"Правильно {correct_amount} / 63: {correct_amount / 0.63 : .1f}%")


if __name__ == "__main__":
    main()
