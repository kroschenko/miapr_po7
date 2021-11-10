from uninn import *
import time
from lab3u import function_lab3_9, repeat_func


def main():
    l1 = LayerSigmoid(lens=(10, 4))
    l2 = LayerLinear(lens=(4, 1))
    nn = NeuralNetwork(l1, l2)

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    times = 30_000
    sep = 1_000
    print(f"- Learning {times} times -")

    start_time = time.time()

    for thousand in range(times // sep):
        for _ in range(sep - 1):
            nn.learn(learn_x, learn_e)
        print(f"{thousand + 1 : 3},000 error: {nn.learn(learn_x, learn_e)}")

    print(f"- Learning time: {time.time() - start_time} seconds -")

    nn.prediction_results_table(test_x, test_e)


if __name__ == "__main__":
    # repeat_func(main)
    main()
