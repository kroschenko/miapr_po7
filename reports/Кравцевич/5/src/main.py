import numpy
import random


class lab5:
	def __init__(self, max_iterations, e, alpha_ki, alpha_ij):
		self.max_iterations = max_iterations
		self.E = e
		self.alpha_ki = alpha_ki
		self.alpha_ij = alpha_ij

		self.etalons = self.get_etalons()
		self.inputs = len(self.etalons[0])
		self.hiddens = self.inputs
		self.outputs = len(self.etalons)

		self.weights_ki = self.get_weights(self.inputs, self.hiddens, -1, 1)
		self.weights_ij = self.get_weights(self.hiddens, self.outputs, -1, 1)

		self.tresholds_i = self.get_thresholds(self.hiddens, -1, 1)
		self.tresholds_j = self.get_thresholds(self.outputs, -1, 1)

	def get_etalons(self):
		vector_1 = numpy.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
		vector_2 = numpy.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
		vector_3 = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

		return numpy.array([vector_1, vector_2, vector_3])

	def get_weights(self, left_neurons, right_neurons, left_border, right_border):
		weights = numpy.zeros((left_neurons, right_neurons))
		for i in range(left_neurons):
			for j in range(right_neurons):
				weights[i][j] = random.uniform(left_border, right_border)
		return weights

	def get_thresholds(self, number_neurons, left_border, right_border):
		tresholds = numpy.zeros(number_neurons)
		for i in range(number_neurons):
			tresholds[i] = random.uniform(left_border, right_border)
		return tresholds

	def learning(self):
		epos = 0
		while epos < self.max_iterations:
			epos += 1
			for etalon in self.etalons:
				x = etalon
				S_i = x.dot(self.weights_ki) - self.tresholds_i
				y_i = numpy.zeros(len(S_i))
				for i in range(len(S_i)):
					y_i[i] = 1. / (1. + numpy.exp(- S_i[i]))

				S_j = y_i.dot(self.weights_ij) - self.tresholds_j
				y_j = S_j
				j_j = numpy.array([y_j[i] - etalon[i] for i in range(len(y_j))])

				dF_j = 1
				j_i = numpy.zeros(self.hiddens)
				for i in range(self.hiddens):
					for j in range(self.outputs):
						j_i[i] += j_j[j] * dF_j * self.weights_ij[i][j]

				for i in range(self.hiddens):
					for j in range(self.outputs):
						self.weights_ij[i][j] -= self.alpha_ij * j_j[j] * dF_j * y_j[j]

				for j in range(self.outputs):
					self.tresholds_j += self.alpha_ij * j_j[j] * dF_j

				for k in range(self.inputs):
					for i in range(self.hiddens):
						dFi = y_i[i] * (1 - y_i[i])
						self.weights_ki[k][i] -= self.alpha_ki * j_i[i] * dFi * y_i[i]
				for i in range(self.hiddens):
					dFi = y_i[i] * (1 - y_i[i])
					self.tresholds_i += self.alpha_ki * j_i[i] * dFi

			E = 0
			for j in range(self.outputs):
				E += 1. / 2 * (y_j[j] - etalon[j]) ** 2
			print(f'epos: {epos} E: {E}')

			if E < self.E or epos >= self.max_iterations:
				print()
				break

		print(f'epos: {epos} E: {E}')
		print('Done')

	def test(self, etalon):
		x = numpy.array(etalon)
		S_i = x.dot(self.weights_ki) - self.tresholds_i
		y_i = numpy.zeros(len(S_i))

		for i in range(len(S_i)):
			y_i[i] = 1. / (1. + numpy.exp(- S_i[i]))

		S_j = y_i.dot(self.weights_ij) - self.tresholds_j
		y_j = S_j

		max_value = y_j[0]
		max_value_index = 0
		for j in range(self.outputs):
			if y_j[j] > max_value:
				max_value = y_j[j]
				max_value_index = j

		print(f'y_j = {y_j}')
		print(f'etalon = {etalon}')
		print(f'etalon is class {max_value_index}')


if __name__ == '__main__':
	obj = lab5(
		5000,
		1e-2,
		0.001,
		0.001
	)
	obj.learning()
	obj.test([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
