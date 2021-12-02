import random
import math
import matplotlib.pyplot as plt

step = 0.1
mistake_min = 0.00001
a, b, c, d = 0.3, 0.5, 0.05, 0.5
numin_nn = 8  # количество входов инс
numin_nel_hidlayer = 3  # количество нэ в скрытом слое
w_ij = [[random.uniform(-0.1, 0.1) for i in range(numin_nel_hidlayer)] for j in range(numin_nn - 1)]
w_jk = [random.uniform(-0.1, 0.1) for _ in range(numin_nel_hidlayer)]
thresholdValue_j = [random.uniform(-0.5, 0.5) for _ in range(numin_nel_hidlayer)]
thresholdValue_k = random.uniform(-0.5, 0.5)
#thresholdValue_j = [0] * numin_nel_hidlayer
#thresholdValue_k = 0


def inputElements(ind):
	y = []
	new_y = []
	for i in range(numin_nn):
		if i == numin_nn - 1:
			y.append(new_y)
			y.append(a * math.cos(b * ind * 0.1) + c * math.sin(d * ind * 0.1))
		else:
			new_y.append(a * math.cos(b * ind * 0.1) + c * math.sin(d * ind * 0.1))
			ind += 1
	return (y)


def hiddenLayer_Sj(j, y):
	S_j = 0
	for i in range(numin_nn - 1):
		S_j += y[0][i] * w_ij[i][j]
	S_j -= thresholdValue_j[j]
	return S_j


def sigmoidActivationFunction(j, y):
	S_j = hiddenLayer_Sj(j, y)
	return (1 / (1 + math.exp(-1 * S_j)))


def outputLayer_Sk(y_j):
	S_k = 0
	for i in range(numin_nel_hidlayer):
		S_k += y_j[i] * w_jk[i]
	S_k -= thresholdValue_k
	return S_k


def change_w_jk(y_j, mistake):
	for i in range(numin_nel_hidlayer):
		w_jk[i] = w_jk[i] - step * mistake * y_j[i]


def change_w_ij(y_j, mistake, y):
	for i in range(numin_nn - 1):
		for j in range(numin_nel_hidlayer):
			w_ij[i][j] = w_ij[i][j] -step * mistake * w_jk[j] * y[0][i] * (y_j[j] * (1 - y_j[j]))


def change_thresholdValue_j(y_j, mistake):
	for i in range(numin_nel_hidlayer):
		thresholdValue_j[i] += step * mistake * y_j[i] * (1 - y_j[i])


def main():
	global thresholdValue_k

	#print("w_jk: ", w_jk)
	#print("w_ij: ", w_ij)
	#print("thresholdValue_j: ", thresholdValue_j)
	#print("thresholdValue_k: ", thresholdValue_k)
	count = 0

	while True:
		for k in range(30):
			y_j, y = [], []
			y_k, error = 0, 0
			y = inputElements(k)
			for i in range(numin_nel_hidlayer):
				y_j.append(sigmoidActivationFunction(i, y))
			y_k = outputLayer_Sk(y_j)
			mistake = y_k - y[-1]
			error += 0.5 * (mistake ** 2)
			change_w_jk(y_j, mistake)
			change_w_ij(y_j, mistake, y)
			change_thresholdValue_j(y_j, mistake)
			thresholdValue_k += mistake * step
		#print("w_jk: ", w_jk)
		#print("w_ij: ", w_ij)
		#print("thresholdValue_j: ", thresholdValue_j)
		#print("thresholdValue_k: ", thresholdValue_k)
		if abs(error) < mistake_min:
			print("finaly: ", error)
			break
		print(count+1,";",error)
		count += 1
	print(count)






if __name__ == '__main__':
	main()

	for i in range(30):

		test_vector_j = []
		test_vector = inputElements(i)
		for j in range(numin_nel_hidlayer):
			test_vector_j.append(sigmoidActivationFunction(j, test_vector))
		prog_result = outputLayer_Sk(test_vector_j)

		right_result = test_vector[-1]

		print('[',i+1,']','Right: ', right_result, '         Program: ', prog_result, '        Error: ', abs(prog_result - right_result))





