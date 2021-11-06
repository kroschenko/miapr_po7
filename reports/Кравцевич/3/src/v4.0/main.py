from layer import Layer
import math

EXPECTED_ERROR = 1e-6
INPUT_SIZE = 6
HIDDEN_LAYER_SIZE = 2
FUNC_STEP = 1

input_layer = Layer(INPUT_SIZE, HIDDEN_LAYER_SIZE)
hidden_layer = Layer(HIDDEN_LAYER_SIZE, 1)


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
	epos = 0
	while True:
		epos += 1
		sum = 0
		for sample in training_data:
			sum += train_model(sample[0], sample[1])

		print(sum)
		if sum < EXPECTED_ERROR:
			print('Number: ', epos)
			break


def train_model(train_sample, expected_result):
	global input_layer
	global hidden_layer

	input_layer.set_input_values(train_sample)
	hidden_layer.set_input_values(input_layer.output_values)

	error = hidden_layer.output_values[0] - expected_result
	errors = input_layer.init_errors(hidden_layer.init_errors([error]))
	# sum = 0
	# for e in errors:
	# 	sum += e ** 2

	return error ** 2


def prediction(sample):
	input_layer.set_input_values(sample)
	hidden_layer.set_input_values(input_layer.output_values)
	return hidden_layer.output_values


if __name__ == '__main__':
	training_data = get_training_data()
	train(get_training_data())

	print('Right: ', training_data[0][1])
	print('Prediction: ', prediction(training_data[0][0]))

	print('Right: ', training_data[1][1])
	print('Prediction: ', prediction(training_data[1][0]))

	print('Right: ', training_data[2][1])
	print('Prediction: ', prediction(training_data[2][0]))
