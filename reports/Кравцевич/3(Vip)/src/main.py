import numpy as np
import math
import random

STEP = 0.1
VECTOR_SIZE = 6
LAYERS_COUNT = 6
EXPECTED_ERROR = 0.000001

weights = [[random.uniform(-1, 1) for _ in range(VECTOR_SIZE)] for _ in range(VECTOR_SIZE)]
T = [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]


def get_func_value(x: float) -> float:
	a, b, d, c = 0.3, 0.1, 0.06, 0.1
	return a * math.cos(b * x) + c * math.sin(d * x)


def get_training_data() -> list:
	data_size = 30
	training_data = []
	current_x = 1

	for _ in range(data_size):
		line = []
		vector = []

		for _ in range(VECTOR_SIZE):
			vector.append(get_func_value(current_x))
			current_x += STEP

		line.append(vector)
		line.append(get_func_value(current_x))
		current_x -= STEP * (VECTOR_SIZE - 1)

		training_data.append(line)

	return training_data

def train_model():
	for line in get_training_data():
		vector, expected = line[0], line[1]

		first_gen = np.array(process_next_vector(vector))
		second_gen = np.array(process_next_vector(first_gen))

		second_gen_error = second_gen - expected

		first_gen_error = 0
		for index, y in enumerate(first_gen):
			first_gen_error += second_gen_error[index] * y * (1 - y) *


def process_next_vector(vector):
	sums = []
	for index, x in enumerate(vector):
		sums.append(process_next_x(index, x))

	y = get_y(sums)

	return y


def process_next_x(index, x):
	w = np.array(weights[index])
	w *= x
	sum = w.sum()
	sum -= T[index]

	return sum


def get_y(sums):
	y = []
	for sum in sums:
		y.append(1 / (1 + math.exp(-sum)))

	return y


if __name__ == '__main__':
	training_data = get_training_data()
	print(*training_data, sep='\n')
	pass
