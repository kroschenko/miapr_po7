import random
import math
from prettytable import PrettyTable
import matplotlib.pyplot as plt

mistake_min = 1e-7
a, b, c, d = 0.3, 0.5, 0.05, 0.5
numin_nn = 8  # количество входов инс
numin_nel_hidlayer = 3  # количество нэ в скрытом слое
w_ij = [[random.uniform(-0.1, 0.1) for i in range(numin_nel_hidlayer)] for j in range(numin_nn - 1)]
w_jk = [random.uniform(-0.1, 0.1) for _ in range(numin_nel_hidlayer)]
thresholdValue_j = [random.uniform(-0.5, 0.5) for _ in range(numin_nel_hidlayer)]
thresholdValue_k = random.uniform(-0.5, 0.5)
step_j, step_k = 1, 1
#thresholdValue_j = [0] * numin_nel_hidlayer
#thresholdValue_k = 0


def inputElements(ind):
	y = []
	new_y = []
	for i in range(numin_nn):
		if i == numin_nn - 1:
			y.append(new_y)
			y.append(a * math.cos(b * ind * 0.3) + c * math.sin(d * ind * 0.3))
		else:
			new_y.append(a * math.cos(b * ind * 0.3) + c * math.sin(d * ind * 0.3))
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


def change_w_jk(y_j, mistake, step_k):
	for i in range(numin_nel_hidlayer):
		w_jk[i] = w_jk[i] - step_k * mistake * y_j[i]


def change_w_ij(y_j, mistake, y, step_j):
	for i in range(numin_nn - 1):
		for j in range(numin_nel_hidlayer):
			w_ij[i][j] = w_ij[i][j] -step_j * mistake * w_jk[j] * y[0][i] * (y_j[j] * (1 - y_j[j]))


def change_thresholdValue_j(y_j, mistake, step_j):
	for i in range(numin_nel_hidlayer):
		thresholdValue_j[i] += step_j * mistake * y_j[i] * (1 - y_j[i])


def new_step_sigmoidActivationFunction(y_j, y_k, mistake):
	numerator, divider, divider_2 = 0, 0, 0
	mistake_j = []
	for i in range(numin_nel_hidlayer):
		mistake_j.append( mistake * y_k * (1 - y_k) * w_jk[i])
	for i in range(numin_nel_hidlayer):
		numerator += (mistake_j[i]**2) * y_j[i]* (1 - y_j[i])
		divider += (mistake_j[i]**2) * (y_j[i]**2)* ((1 - y_j[i])**2)
	step_j = (4 * numerator) / ((1 + y_k**2) * divider)
	return step_j




def main():
	global thresholdValue_k
	global step_k
	global step_j


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
			step_j = new_step_sigmoidActivationFunction(y_j, y_k, mistake)
			step_k = 1 / (1 + (y_j[0] ** 2) + (y_j[1] ** 2) + (y_j[2] ** 2))
			change_w_jk(y_j, mistake, step_k)
			change_w_ij(y_j, mistake, y, step_j)
			change_thresholdValue_j(y_j, mistake, step_j)
			thresholdValue_k += mistake * step_k


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
	mytable = PrettyTable()
	mytable.field_names = ["Right:", "Program:", "Error:"]
	x, y = [], []
	for i in range(30):

		x.append(i)
		test_vector_j = []
		test_vector = inputElements(i)
		for j in range(numin_nel_hidlayer):
			test_vector_j.append(sigmoidActivationFunction(j, test_vector))
		prog_result = outputLayer_Sk(test_vector_j)

		right_result = test_vector[-1]
		y.append(abs(prog_result - right_result))
		mytable.add_row([right_result, prog_result, abs(prog_result - right_result)])
	plt.plot(x, y)
	plt.show()
	print(mytable)



