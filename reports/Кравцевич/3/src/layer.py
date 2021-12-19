import random


class Layer:

	def __init__(self, input_sample_size: int, next_layer_size: int, activation_func):
		self.step = 1
		self.weights = []
		self.T = []
		self.errors = []
		self.input_values = []
		self.output_values = []
		self.next_layer_size = next_layer_size
		self.weights = [[random.uniform(0, 1) for _ in range(next_layer_size)] for __ in range(input_sample_size)]
		self.T = [random.uniform(0, 1) for _ in range(next_layer_size)]

		self.activation_func = activation_func

	def set_input_values(self, input_values: list) -> None:
		self.input_values = input_values
		self.output_values = self.get_output_values()

	def get_layer_sums(self) -> list:
		sum_list = []

		for next_node_index in range(self.next_layer_size):
			sum = 0
			for node_index in range(len(self.input_values)):
				sum += self.input_values[node_index] * self.weights[node_index][next_node_index]
			sum -= self.T[next_node_index]
			sum_list.append(sum)
		return sum_list

	def get_output_values(self):
		sums = self.get_layer_sums()
		return self.activation_func.get_output_values(sums)

	def init_errors(self, error: list) -> list:
		errors = []

		for node_weights in self.weights:
			error_sum = 0
			for index, weight in enumerate(node_weights):
				error_sum += error[index] * self.output_values[index] * (1 - self.output_values[index]) * weight
			errors.append(error_sum)

		self.change_weights(error)
		return errors

	def change_weights(self, errors: list) -> None:
		self.weights, self.T = self.activation_func.get_new_weights(self.weights, self.T, self.input_values,
																	self.output_values, errors)
