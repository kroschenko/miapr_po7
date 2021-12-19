from layer import *

vectors = [[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]]

byte = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]

INPUT_SIZE = len(vectors[0])
HIDDEN_SIZE = 10
OUTPUT_SIZE = len(vectors)
E = 1e-20

input_layer = Layer(INPUT_SIZE, HIDDEN_SIZE)
hide_layer = LastLayer(HIDDEN_SIZE, OUTPUT_SIZE)


def predict(vector):
    return hide_layer.set_values(input_layer.set_values(vector))


def education():
    error = E
    errors = []
    while abs(error) >= E:
        error = 0
        for ind, vector in enumerate(vectors):
            e = predict(vector) - np.array(byte[ind])
            error = sum([1. / 2 * (er ** 2) for er in e])
            hide_layer.weights_change(e)
            input_layer.weights_change(hide_layer.error)
        print(error)


if __name__ == '__main__':
    education()
    noisyVector = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    result = predict(noisyVector)
    print('vector with noise: ', noisyVector)
    print('vector in class:   ', vectors[np.where(max(result) == result)[0][0]])
