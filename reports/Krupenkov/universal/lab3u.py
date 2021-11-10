from uninn import *
import time


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def repeat_func(func):
    while True:
        func()
        ans = input("Еще? (y/n): ")
        if not ans or (ans[0] != "y" and ans != "н"):
            break


def main():
    l1 = Layer(lens=(10, 4), f_act=funsact.sigmoid, d_f_act=funsact.d_sigmoid)
    l2 = Layer(lens=(4, 1))
    nn = NeuralNetwork(l1, l2)

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    alpha = 0.06
    times = 30_000
    sep = 1_000
    print(f"- Learning {times} times -")

    start_time = time.time()

    for thousand in range(times // sep):
        for _ in range(sep - 1):
            nn.learn(learn_x, learn_e, alpha)
        print(f"{thousand + 1 : 3},000 error: {nn.learn(learn_x, learn_e, alpha)}")

    print(f"- Learning time: {time.time() - start_time} seconds -")

    nn.prediction_results_table(test_x, test_e)


if __name__ == "__main__":
    # repeat_func(main)
    main()
