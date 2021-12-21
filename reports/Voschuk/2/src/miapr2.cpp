#include <iostream>
#include <iomanip>
#include <ctime>
#include <windows.h>
using namespace std;

int main() {
	SetConsoleOutputCP(1251);
	SetConsoleCP(1251);
	int a = 2,
		b = 6,
		enteries = 4, //входы ИНС
		n = 30, //количество значений для обучения
		values = 15; //количество значений для прогнозирования
	double d = 0.2,
		Em = 0.05, //минимальная среднеквадратичная ошибка сети
		E, //суммарная среднеквадратичная ошибка сети
		T = 1; //порог НС
	double* W = new double[enteries]; //весовые коэффициенты (3)
	//srand(time(NULL)); //для разного рандома
	for (int i = 0; i < enteries; i++) { //генерирует весовые коэффициенты
		W[i] = (double)(rand()) / RAND_MAX; //от 0 до 1
		cout << "W[" << i << "] = " << W[i] << endl; //вывод весовых коэффициентов
	}
	cout << endl;
	double* etalon_values = new double[n + values]; //эталонные значения y
	for (int i = 0; i < n + values; i++) { //вычисляем эталонные значения
		double step = 0.1; //шаг
		double x = step * i;
		etalon_values[i] = a * sin(b * x) + d; //формула для проверки
	}
	int era = 0; //для индексов
	while (1) {
		double y1; //выходное значение нейронной сети
		double Alpha = 0.05; //скорость обучения
		E = 0; //ошибка
		for (int i = 0; i < n - enteries; i++) {
			y1 = 0;
			double temp = 0.0;
			for (int j = 0; j < enteries; j++) {
				temp += pow(etalon_values[i + j], 2);
			}
			Alpha = 1 / (1 + temp); //адаптивный шаг
			for (int j = 0; j < enteries; j++) { //векторы выходной активности сети
				y1 += W[j] * etalon_values[j + i];
			}
			y1 -= T;
			for (int j = 0; j < enteries; j++) { //изменение весовых коэффициентов
				W[j] -= Alpha * (y1 - etalon_values[i + enteries]) * etalon_values[i + j];
			}
			T += Alpha * (y1 - etalon_values[i + enteries]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - etalon_values[i + enteries], 2); //расчет суммарной среднеквадратичной ошибки
		}
		era++;
		cout << era << " | " << E << endl;
		if (E < Em) break;
	} //далее сеть обучена
	cout << endl;
	cout << "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ" << endl;
	cout << setw(27) << right << "Эталонные значения" << setw(23) << right << "Полученные значения";
	cout << setw(23) << right << "Отклонение" << endl;
	double* prognoz_values = new double[n + values];
	for (int i = 0; i < n; i++) {
		prognoz_values[i] = 0;
		for (int j = 0; j < enteries; j++) {
			prognoz_values[i] += W[j] * etalon_values[j + i]; //получаемые значения в результате обучения
		}
		prognoz_values[i] -= T;
		cout << "y[" << i + 1 << "] = " << setw(20) << right << etalon_values[i + enteries] << setw(23) << right;
		cout << prognoz_values[i] << setw(23) << right << etalon_values[i + enteries] - prognoz_values[i] << endl;
	}
	cout << endl << "РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ" << endl;
	cout << setw(28) << right << "Эталонные значения" << setw(23) << right << "Полученные значения" << setw(23) << right << "Отклонение" << endl;
	for (int i = 0; i < values; i++) {
		prognoz_values[i + n] = 0;
		for (int j = 0; j < enteries; j++) {
			//прогнозируемые значения
			prognoz_values[i + n] += W[j] * etalon_values[n - enteries + j + i];
		}
		prognoz_values[i + n] -= T;
		cout << "y[" << n + i + 1 << "] = " << setw(20) << right << etalon_values[i + n] << setw(23) << right;
		cout << prognoz_values[i + n] << setw(23) << right << etalon_values[i + n] - prognoz_values[i + n] << endl;
	}
	delete[]etalon_values;
	delete[]prognoz_values;
	delete[]W;
	system("pause");
	return 0;
}