import activation_functions
from layer import Layer
import math
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_ERROR = 1e-6
INPUT_SIZE = 6
HIDDEN_LAYER_SIZE = 2
FUNC_STEP = 0.5

input_layer = Layer(INPUT_SIZE, HIDDEN_LAYER_SIZE, activation_functions.Sigmoid())
hidden_layer = Layer(HIDDEN_LAYER_SIZE, 1, activation_functions.Linear())


def get_func_value(x):
	return 0.3 * math.cos(0.1 * x) + 0.06 * math.sin(0.1 * x)


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


def train(training_data):
	epoch = 0
	error_points = []

	while True:
		epoch += 1
		sum = 0
		for sample in training_data:
			sum += train_model(sample[0], sample[1])

		error_points.append(sum)
		print(sum)

		if sum < EXPECTED_ERROR:
			error_points = error_points[:50]
			plt.plot(range(len(error_points)), error_points)
			plt.savefig('errors')
			print('Epoch: ', epoch)
			break


def train_model(train_sample, expected_result):
	input_layer.set_input_values(train_sample)
	hidden_layer.set_input_values(input_layer.output_values)

	error = hidden_layer.output_values[0] - expected_result
	input_layer.init_errors(hidden_layer.init_errors([error]))

	return error ** 2


def prediction(sample):
	input_layer.set_input_values(sample)
	hidden_layer.set_input_values(input_layer.output_values)

	return hidden_layer.output_values[0]


if __name__ == '__main__':
	training_data = get_training_data()
	train(get_training_data())

	right_results, results, errors = [], [], []

	for line in training_data:
		test_vector = line[0]
		result = line[1]
		predict = prediction(test_vector)

		right_results.append(result)
		results.append(predict)
		errors.append(abs(result - predict))

	table = pd.DataFrame({
		'Right': pd.Series(right_results),
		'Received': pd.Series(results),
		'Errors': pd.Series(errors)
	})

	print(table)
