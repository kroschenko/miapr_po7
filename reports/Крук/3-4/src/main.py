from layer import *
import matplotlib.pyplot as plt

INPUT_SIZE = 6
HIDDEN_SIZE = 2
EXPRESSION_STEP = 0.2
E = 1e-6

input_layer = Layer(INPUT_SIZE, HIDDEN_SIZE)
hide_layer = LastLayer(HIDDEN_SIZE)


def expression(x):
    a, b, c, d = 0.4, 0.2, 0.07, 0.2
    return a * math.cos(b * x) + c * math.sin(d * x)


def training_data():
    vectors = []
    start_x = 0
    for _ in range(15):
        vector = []
        for i in range(INPUT_SIZE):
            vector.append(expression(start_x + i * EXPRESSION_STEP))
        vectors.append([vector, expression(start_x + INPUT_SIZE * EXPRESSION_STEP)])
        start_x += EXPRESSION_STEP
    return vectors


def predict(vector):
    input_layer.set_values(vector)
    hide_layer.set_values(input_layer.output_values)
    return hide_layer.result


def education():
    train_data = training_data()
    error = E
    errors = []
    while abs(error) >= E:
        error = 0
        for data in train_data:
            e = predict(data[0]) - data[1]
            error += e ** 2 / 2
            hide_layer.weights_change(e)
            input_layer.weights_change(hide_layer.error)
        print(error)
        errors.append(error)
    plt.plot([x for x in range(len(errors))], errors)
    plt.show()


if __name__ == '__main__':
    education()
    train_data = training_data()
    for data in train_data:
        print('{0:<28}{1:<10}{2:<30}'.format(predict(data[0]), '|', data[1]))
