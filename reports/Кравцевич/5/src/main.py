import numpy
import random

max_iterations = 5000
E = 1e-5
alpha_ki = 0.001
alpha_ij = 0.001


def get_etalons():
	vector_1 = numpy.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
	vector_2 = numpy.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
	vector_3 = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

	return numpy.array([vector_1, vector_2, vector_3])


def get_weights(left_neurons, right_neurons, left_border, right_border):
	weights = numpy.zeros((left_neurons, right_neurons))
	for i in range(left_neurons):
		for j in range(right_neurons):
			weights[i][j] = random.uniform(left_border, right_border)
	return weights


def get_thresholds(number_neurons, left_border, right_border):
	tresholds = numpy.zeros(number_neurons)
	for i in range(number_neurons):
		tresholds[i] = random.uniform(left_border, right_border)
	return tresholds


etalons = get_etalons()
inputs = len(etalons[0])
hiddens = inputs
outputs = len(etalons)

weights_ki = get_weights(inputs, hiddens, -1, 1)
weights_ij = get_weights(hiddens, outputs, -1, 1)

tresholds_i = get_thresholds(hiddens, -1, 1)
tresholds_j = get_thresholds(outputs, -1, 1)


def learning():
	global tresholds_i
	global tresholds_j
	epos = 0
	while epos < max_iterations:
		epos += 1
		for etalon in etalons:
			x = etalon
			S_i = x.dot(weights_ki) - tresholds_i
			y_i = numpy.zeros(len(S_i))
			for i in range(len(S_i)):
				y_i[i] = 1. / (1. + numpy.exp(- S_i[i]))

			S_j = y_i.dot(weights_ij) - tresholds_j
			y_j = S_j
			j_j = numpy.array([y_j[i] - etalon[i] for i in range(len(y_j))])

			dF_j = 1
			j_i = numpy.zeros(hiddens)
			for i in range(hiddens):
				for j in range(outputs):
					j_i[i] += j_j[j] * dF_j * weights_ij[i][j]

			for i in range(hiddens):
				for j in range(outputs):
					weights_ij[i][j] -= alpha_ij * j_j[j] * dF_j * y_j[j]

			for j in range(outputs):
				tresholds_j += alpha_ij * j_j[j] * dF_j

			for k in range(inputs):
				for i in range(hiddens):
					dFi = y_i[i] * (1 - y_i[i])
					weights_ki[k][i] -= alpha_ki * j_i[i] * dFi * y_i[i]
			for i in range(hiddens):
				dFi = y_i[i] * (1 - y_i[i])
				tresholds_i += alpha_ki * j_i[i] * dFi

		Error = 0
		for j in range(outputs):
			Error += 1. / 2 * (y_j[j] - etalon[j]) ** 2
		print(f'epos: {epos} E: {Error}')

		if Error < E or epos >= max_iterations:
			print()
			break

	print(f'epos: {epos} E: {Error}')
	print('Done')


def test(etalon):
	x = numpy.array(etalon)
	S_i = x.dot(weights_ki) - tresholds_i
	y_i = numpy.zeros(len(S_i))

	for i in range(len(S_i)):
		y_i[i] = 1. / (1. + numpy.exp(- S_i[i]))

	S_j = y_i.dot(weights_ij) - tresholds_j
	y_j = S_j

	max_value = y_j[0]
	max_value_index = 0
	for j in range(outputs):
		if y_j[j] > max_value:
			max_value = y_j[j]
			max_value_index = j

	print(f'y_j = {y_j}')
	print(f'etalon = {etalon}')
	print(f'etalon is class {max_value_index}')


if __name__ == '__main__':
	learning()
	test([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
