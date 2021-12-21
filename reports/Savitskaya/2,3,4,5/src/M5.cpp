#include <iostream>
#include <math.h>
#define e 2.71828
using namespace std;

double sigmoid(double x) {  //Сигмоид
	return 1 / (1 + pow(e, -x));
}
//работа со скрытым слоем
double* get_hiddens(bool* Inputs, double w12[6][2], double T_Hid[]) {
	double* Hiddens = new double[2];
	for (int i = 0; i < 2; i++) Hiddens[i] = 0;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 6; j++) {
			Hiddens[i] += w12[j][i] * Inputs[j];
		}
		Hiddens[i] -= T_Hid[i];
		Hiddens[i] = sigmoid(Hiddens[i]);
	}
	return Hiddens;
}
//получение результатов
double* get_result(bool* Inputs, double w12[6][2], double T_Hid[], double w23[2][1], double T_Out[], double Hiddens[2]) {
	double* Results = new double[1];
	for (int i = 0; i < 1; i++)
		Results[i] = 0;
	for (int j = 0; j < 1; j++) {
		for (int i = 0; i < 2; i++) {
			Results[j] += Hiddens[i] * w23[i][j];
		}
		Results[j] -= T_Out[j];
		Results[j] = sigmoid(Results[j]);
	}
	return Results;
}


int main() {
	system("color f0");
	setlocale(0, "");
	///Векторы согласно варианту
	bool Vector1[] = { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 };
	bool Vector2[] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0 };
	bool Vector3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 };
	///ввод входных сигналов и векторов
	bool* Inputs = new bool[6];
	for (int i = 0; i < 6; i++) Inputs[i] = 0;
	bool** Vectors = new bool* [8];
	Vectors[0] = Vector1;
	Vectors[1] = Vector2;
	Vectors[2] = Vector3;
	///
	double w12[6][2], w23[2][1], T_Hid[2], T_Out[1], E_min = 0.0001, alpha = 0.04, Ethalon, E = 0, Outputs[1] = { 0 };
	double* Currents = new double[1];
	double* Hiddens = new double[2];
	double Mistakes[1] = { 0 };
	double Ethalons[1] = { 0 };
	double MistakesHid[2] = { 0 };
	int Iter = 1;  //начальное количество итераций
	///рандомизация весов и порогов
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 2; j++) {
			w12[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < 1; k++) {
				w23[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				T_Out[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			T_Hid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	int H = 0;
	do {
		E = 0;
		for (int N = 0; N < 1; N++) {
			Ethalons[0] = 0;
			Ethalons[N] = 1;
			for (int q = 0; q < Iter; q++) {
				///нахождение результатов и ошибок
				Inputs = Vectors[N];
				Hiddens = get_hiddens(Inputs, w12, T_Hid);
				Currents = get_result(Inputs, w12, T_Hid, w23, T_Out, Hiddens);
				for (int i = 0; i < 1; i++)
					Mistakes[i] = Currents[i] - Ethalons[i];
				///ошибки на скрытом слое
				for (int j = 0; j < 1; j++) {
					for (int m = 0; m < 1; m++) {
						MistakesHid[j] += Mistakes[m] * Currents[m] * (1 - Currents[m]) * w23[j][m];
					}
				}

				for (int j = 0; j < 1; j++) {
					for (int i = 0; i < 2; i++) {
						w23[i][j] -= alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]) * Hiddens[i];
					}
					T_Out[j] += alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]);
				}

				for (int j = 0; j < 2; j++) {
					for (int i = 0; i < 6; i++) {
						w12[i][j] -= alpha * MistakesHid[j] * Hiddens[j] * (1 - Hiddens[j]) * Inputs[i];
					}
					T_Hid[j] += alpha * MistakesHid[j] * Hiddens[j] * (1 - Hiddens[j]);
				}
				E += pow(Mistakes[N], 2);
			}
		}
		E /= 2;
		if (H % 100 == 0 || H < 300) {
			cout << H << ";" << E << endl;
		}
		H++;

	} while (E > E_min);

	//прогнозирование
	double* HiddenPred;
	double* Values;
	bool Vectors3[] = { 0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,0,0,0 };
	bool Vectors4[] = { 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0 };
	bool Vectors5[] = { 1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	bool Vectors6[] = { 1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vectors7[] = { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 };
	Vectors[3] = Vectors3;
	Vectors[4] = Vectors4;
	Vectors[5] = Vectors5;
	Vectors[6] = Vectors6;
	Vectors[7] = Vectors7;

	for (int i = 0; i < 8; i++) {
		Inputs = Vectors[i];
		cout << "Результат вектора " << i + 1 << " значение которого ";
		for (int j = 0; j < 20; j++) {
			cout << Inputs[j] << ' ';
		}
		cout << endl << "Результат : ";
		HiddenPred = get_hiddens(Inputs, w12, T_Hid);
		Values = get_result(Inputs, w12, T_Hid, w23, T_Out, HiddenPred);
		cout << Values[0] << endl;
	}
	system("pause");
}
