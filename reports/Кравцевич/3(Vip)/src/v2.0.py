import random
import math

import numpy as np

STEP = 0.1
LAYOUT_COUNT = 2
VECTOR_SIZE = 6

weights = {
	'l1': [[random.uniform(-1, 1) for _ in range(VECTOR_SIZE)] for _ in range(VECTOR_SIZE)],
	'l2': [[random.uniform(-1, 1) for _ in range(VECTOR_SIZE)] for _ in range(VECTOR_SIZE)]
}

T = {
	'l1': random.uniform(-1, 1),
	'l2': random.uniform(-1, 1)
}


def get_func_value(x):
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
	training_data = get_training_data()

	for line in training_data:
		vector, expected = line[0], line[1]

		# Слой 1
		sums1 = get_sums(vector, 1)
		y1 = get_y(sums1)

		# Слой 2
		sums2 = get_sums(y1, 2)
		y2 = get_y(sums2)


def get_sums(vector, layout_num):
	sums = [0] * len(vector)
	for m in range(VECTOR_SIZE):
		for n in range(VECTOR_SIZE):
			sums[m] += weights[f'l{layout_num}'][m][n] * vector[n]

	return sums


def get_y(sums):
	y = []
	for sum in sums:
		y.append(1 / (1 + math.exp(-sum)))

	return y
