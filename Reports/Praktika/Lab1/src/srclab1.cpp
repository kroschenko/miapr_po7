#include <iostream>
#include <iomanip>
using namespace std;

int main() {
	setlocale(LC_ALL, "rus");
	cout << "Среднеквадратичная ошибка : " << endl;
	int a = 1,
		b = 9,
		num_enteries = 4, //количество входов ИНС
		n = 30, //количесвто значений, на которых производится обучение
		num_predicated_values = 15; //количесвто значений, на которых производится прогнозирование

	double d = 0.5,
		Em = 0.01, //минимальная среднеквадратичная ошибка сети
		E, //суммарная среднеквадратичная ошибка
		T = 1; //порог нейронной сети

	double* W = new double[num_enteries]; //весовые коэффициенты
	for (int i = 0; i < num_enteries; i++) { //задаем случайным образом весовые коэффициенты
		W[i] = static_cast <double> (rand()) / (static_cast <double>(RAND_MAX / 10));
	}

	double* reference_value_y = new double[n + num_predicated_values]; //эталонные значения y
	for (int i = 0; i < n + num_predicated_values; i++) { //вычисляем эталонные значения
		double step = 0.1, //шаг
			x = step * i;
		reference_value_y[i] = a * sin(b * x) + d;
	}

	do {
		double y1, //выходное значение нейронной сети
			A = 0.001; //скорость обучения
		E = 0;

		for (int i = 0; i < n - num_enteries; i++) {
			y1 = 0;

			for (int j = 0; j < num_enteries; j++) { //векторы выходной активности сети
				y1 += W[j] * reference_value_y[i + j];
			}
			y1 -= T;

			for (int j = 0; j < num_enteries; j++) { //изменение весовых коэффициентов
				W[j] -= A * (y1 - reference_value_y[i + num_enteries]) * reference_value_y[i + j];
			}

			T += A * (y1 - reference_value_y[i + num_enteries]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - reference_value_y[i + num_enteries], 2); //расчет суммарной среднеквадратичной ошибки
		}
		cout << E << endl;
	} while (E > Em);

	cout << "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ" << endl;
	cout << setw(27) << left << "Эталонные значения" << setw(23) << left << "Полученные значения" << "Отклонение" << endl; cout << "Среднеквадратичная ошибка" << endl;
	double* predicated_values = new double[n + num_predicated_values];

	for (int i = 0; i < n; i++) {
		predicated_values[i] = 0;
		for (int j = 0; j < num_enteries; j++) {
			predicated_values[i] += W[j] * reference_value_y[j + i] - num_enteries;//получаемые значения в результате обучения
		}
		predicated_values[i] -= T;

		cout << "y[" << i << "] = " << setw(20) << left << reference_value_y[i] << setw(23) << left;
		cout << predicated_values[i] << reference_value_y[i] - predicated_values[i] << endl;
	}

	cout << endl << "РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ" << endl;
	cout << setw(28) << left << "Эталонные значения" << setw(23) << left << "Полученные значения" << "Отклонение" << endl;

	for (int i = 0; i < num_predicated_values; i++) {
		predicated_values[i + n] = 0;
		for (int j = 0; j < num_enteries; j++) {
			//прогнозируемые значения
			predicated_values[i + n] += W[j] * reference_value_y[n - num_enteries + j + i];
		}
		predicated_values[i + n] += T;

		cout << "y[" << n + i << "] = " << setw(20) << left << reference_value_y[i + n] << setw(23) << left;
		cout << predicated_values[i + n] << reference_value_y[i + n] - predicated_values[i + n] << endl;
	}

	delete[]reference_value_y;
	delete[]predicated_values;
	delete[]W;

	system("pause");
	return 0;

}