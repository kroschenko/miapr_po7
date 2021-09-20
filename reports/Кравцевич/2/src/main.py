import math
import random

VECTOR_SIZE = 3
EXPECTED_ERROR = 0.0000001

weights = [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]
T = random.uniform(-1, 1)


def get_func_value(x: float) -> float:
	a, b, d = 3, 6, 0.1
	return a * math.sin(b * x) + d


def get_training_data() -> list:
	data_size = 30
	training_data = []
	current_x = 1
	step = 0.1
	for _ in range(data_size):
		line = []
		vector = []

		for _ in range(VECTOR_SIZE):
			vector.append(get_func_value(current_x))
			current_x += step

		line.append(vector)
		line.append(get_func_value(current_x))
		current_x -= step * 2

		training_data.append(line)

	return training_data


def train_model() -> None:
	common_error = 0
	iteration = 0
	while True:
		for line in get_training_data():
			vector = line[0]
			expected_result = line[-1]
			obtained_result = predict(vector)

			error = obtained_result - expected_result
			common_error += error ** 2

			change_weights(error, vector)

		if math.fabs(common_error) <= EXPECTED_ERROR:
			print('Iteration: ', iteration)
			break

		iteration += 1
		common_error = 0


def predict(vector: list) -> float:
	output = 0

	for index, value in enumerate(vector):
		output += value * weights[index]

	output -= T
	return output


def change_weights(error: float, vector: list) -> None:
	global T, weights

	step = get_step(vector)
	for w_index in range(len(weights)):
		weights[w_index] -= step * error * vector[w_index]

	T += step * error


def get_step(vector):
	return 1 / (1 + sum([x ** 2 for x in vector]))


if __name__ == '__main__':
	print(*get_training_data(), sep='\n')
	train_model()
	print('The training is complete!')

	test_vector = [get_func_value(1 + 0.1 * n) for n in range(3)]
	right_answer = get_func_value(1 + 0.1 * 3)
	print('Test vector:', *test_vector, sep=' ')
	print('Obtained result: ', predict(test_vector))
	print('Right result: ', right_answer)
