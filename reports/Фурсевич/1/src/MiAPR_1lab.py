import random
import math

step, speed = 0.1, 0.1
mistake_min = 0.001
a, b, d = 3, 5, 0.5
kol_vh = 4
limit = 0
w = [random.uniform(-0.1, 0.1) for _ in range(3)]


def row_elements(ind):
	y = []
	new_y = []
	for i in range(kol_vh):
		if i == kol_vh - 1:
			y.append(new_y)
			y.append(a * math.sin(b * (ind * 0.1)) + d)
		else:
			new_y.append(a * math.sin(b * (ind * 0.1)) + d)
			ind += 1
	return (y)


if __name__ == '__main__':
	error = 0
	while True:
		for i in range(45):
			y = row_elements(i)
			y_pract = y[0][0] * w[0] + y[0][1] * w[1] + y[0][2] * w[2] - limit
			mistake = y_pract - y[1]
			error += 0.5 * (mistake ** 2)
			limit += speed * mistake
			for j in range(kol_vh - 1):
				w[j] -= speed * mistake * y[0][j]
		# print(error)
		if error <= mistake_min:
			break
		error = 0

	for i in range(45):
		if i == 0:
			print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
			print("Эталонные значения",'  ', "Полученные значения",'  ', "Отклонение")
		elif i == 30:
			print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
			print("Эталонные значения",'  ', "Полученные значения",'  ', "Отклонение")

		y = row_elements(i)
		y_pract = y[0][0] * w[0] + y[0][1] * w[1] + y[0][2] * w[2] - limit
		mistake = y_pract - y[1]
		print(y[-1],'  ', y_pract,'  ', abs(mistake))
		error += 0.5 * (mistake ** 2)
		limit += speed * mistake
		for j in range(kol_vh - 1):
			w[j] -= speed * mistake * y[0][j]
