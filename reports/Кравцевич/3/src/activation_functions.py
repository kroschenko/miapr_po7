import math

DEFAULT_STEP = 1


class ActivationFunc:
	def get_output_values(self, sums):
		pass

	def get_new_weights(self, weights, T, input_values, output_values, errors):
		pass


class Sigmoid(ActivationFunc):
	def __init__(self, step=DEFAULT_STEP):
		self.training_step = step

	def get_output_values(self, sums):
		return [1 / (1 + math.exp(-s)) for s in sums]

	def get_new_weights(self, weights, T, input_values, output_values, errors):
		for node_index, node_weights in enumerate(weights):
			for next_node_index, weight_value in enumerate(node_weights):
				weights[node_index][next_node_index] -= self.training_step * errors[next_node_index] * output_values[
					next_node_index] * (1 - output_values[next_node_index]) * input_values[node_index]

		for t_index, t in enumerate(T):
			T[t_index] += self.training_step * errors[t_index] * output_values[t_index] * (
					1 - output_values[t_index])

		return weights, T


class Linear(ActivationFunc):
	def __init__(self, step=DEFAULT_STEP):
		self.training_step = step

	def get_output_values(self, sums):
		return sums

	def get_new_weights(self, weights, T, input_values, output_values, errors):
		for node_index, node_weights in enumerate(weights):
			for next_node_index, weight_value in enumerate(node_weights):
				weights[node_index][next_node_index] -= self.training_step * input_values[node_index] * errors[next_node_index]

		for t_index, t in enumerate(T):
			T[t_index] += self.training_step * errors[t_index]

		return weights, T
