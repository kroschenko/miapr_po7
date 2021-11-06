import math
import random
import pandas as pd
import matplotlib.pyplot as plt

EXCPECTED_ERROR = 1e-5
INPUT_SIZE = 6
HIDDEN_LAYER_SIZE = 2
FUNC_STEP = 0.1
TRAINING_STEP = 0.5
weights = {}
T = {}

error_changes = []


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

		for _ in range(INPUT_SIZE):
			vector.append(get_func_value(current_x))
			current_x += FUNC_STEP

		line.append(vector)
		line.append(get_func_value(current_x))
		current_x -= FUNC_STEP * (INPUT_SIZE - 1)

		training_data.append(line)
	return training_data


def train_model(training_data):
	sum_error = 1

	while (abs(sum_error) > EXCPECTED_ERROR):
		sum_error = 0
		for sample in training_data:
			data = sample[0]
			expected_result = sample[1]

			results = {}
			results[0] = data  # input data
			results[1] = get_layer_value(data)  # hidden layer
			results[2] = get_layer_value(results[1], 2, 1)[0]  # output layer

			error = results[2] - expected_result

			prev_errors = []
			for layer_number in range(2, 0, -1):
				for y_index in range(len(results[layer_number - 1])):
					prev_error = error * results[layer_number - 1][y_index] * (1 - results[layer_number - 1][y_index]) * \
								 weights[layer_number][y_index][0]
					prev_errors.append(prev_error)
				sum_error += 0.5 * sum([x ** 2 for x in prev_errors])
				change_weights(prev_errors, layer_number, results[layer_number], results[layer_number - 1])

		error_changes.append(sum_error)
		print(sum_error)


def change_weights(errors, layer_number, values, current):
	global TRAINING_STEP
	if type(errors) != list:
		errors = [errors]
	if type(values) != list:
		values = [values]

	for index, weight in enumerate(weights[layer_number]):
		for i, w in enumerate(weight):
			weights[layer_number][index][i] -= TRAINING_STEP * errors[i] * values[i] * (1 - values[i]) * current[index]

			T[layer_number][i] += TRAINING_STEP * errors[i] * values[i] * (1 - values[i])


def get_layer_value(input_data, layer_number=1, layer_size=HIDDEN_LAYER_SIZE):
	if layer_number not in weights:
		init_weights(layer_number, len(input_data), layer_size)

	sums = [get_sum(input_data, layer_number, i + 1, ) for i in range(layer_size)]

	values = get_values(sums)

	return values


def get_values(sums):
	output = []
	for S in sums:
		output.append(1 / (1 + math.exp(-1 * S)))

	return output


def get_sum(input_data, layer_number, sum_number):
	sum = 0

	for i in range(len(input_data)):
		sum += input_data[i] * weights[layer_number][i][sum_number - 1]
	sum -= T[layer_number][sum_number - 1]

	return sum


def init_weights(number, input_size, layer_size):
	weight_matrix = []

	for _ in range(input_size):
		weight_matrix.append([random.uniform(0, 1) for __ in range(layer_size)])
	t = [random.uniform(0, 1) for _ in range(layer_size)]

	weights[number] = weight_matrix
	T[number] = t


def predict(vector):
	return get_layer_value(get_layer_value(vector), 2, 1)[0]


if __name__ == '__main__':
	training_data = get_training_data()
	train_model(training_data)

	right_results, results, errors = [], [], []

	for line in training_data:
		test_vector = line[0]
		result = line[1]
		prediction = predict(test_vector)

		right_results.append(result)
		results.append(prediction)
		errors.append(abs(result - prediction))

	table = pd.DataFrame({
		'Right': pd.Series(right_results),
		'Received': pd.Series(results),
		'Errors': pd.Series(errors)
	})
