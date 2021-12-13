#include <iostream>
#include <math.h>
#define NUM_IN 15
#define NUM_HID 4
#define NUM_OUT 1
#define e 2.71828
using namespace std;

double sigmoid(double x) {
	return 1 / (1 + pow(e, -x));
}

double* get_hiddens(bool* Inputs, double w12[NUM_IN][NUM_HID], double T_Hid[]) {
	double* Hiddens = new double[NUM_HID];
	for (int i = 0; i < NUM_HID; i++) Hiddens[i] = 0;
	for (int i = 0; i < NUM_HID; i++) {
		for (int j = 0; j < NUM_IN; j++) {
			Hiddens[i] += w12[j][i] * Inputs[j];
		}
		Hiddens[i] -= T_Hid[i];
		Hiddens[i] = sigmoid(Hiddens[i]);
	}
	return Hiddens;
}

double* get_result(bool* Inputs, double w12[NUM_IN][NUM_HID], double T_Hid[], double w23[NUM_HID][NUM_OUT], double T_Out[], double Hiddens[NUM_HID]) {
	double* Results = new double[NUM_OUT];
	for (int i = 0; i < NUM_OUT; i++)
		Results[i] = 0;
	for (int j = 0; j < NUM_OUT; j++) {
		for (int i = 0; i < NUM_HID; i++) {
			Results[j] += Hiddens[i] * w23[i][j];
		}
		Results[j] -= T_Out[j];
		Results[j] = sigmoid(Results[j]);
	}
	return Results;
}


int main() {
	bool Vect1[] = { 0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,0,0,0 };
	bool Vect2[] = { 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0 };
	bool Vect3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 };
	bool* Inputs = new bool[NUM_IN];
	for (int i = 0; i < NUM_IN; i++) Inputs[i] = 0;
	bool** Vectors = new bool* [8];
	Vectors[0] = Vect1;
	Vectors[1] = Vect2;
	Vectors[2] = Vect3;
	double w12[NUM_IN][NUM_HID], w23[NUM_HID][NUM_OUT], T_Hid[NUM_HID], T_Out[NUM_OUT], E_min = 0.0001, alpha = 0.04, Ethalon, E = 0, Outputs[NUM_OUT] = { 0 };
	double* Currents = new double[NUM_OUT];
	double* Hiddens = new double[NUM_HID];
	double Mistakes[NUM_OUT] = { 0 };
	double Ethalons[NUM_OUT] = { 0 };
	double MistakesHid[NUM_HID] = { 0 };
	int Iter = 1;
	for (int i = 0; i < NUM_IN; i++) {
		for (int j = 0; j < NUM_HID; j++) {
			w12[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < NUM_OUT; k++) {
				w23[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				T_Out[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			T_Hid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	int H = 0;
	do {
		E = 0;
		for (int N = 0; N < NUM_OUT; N++) {
			Ethalons[0] = 0;
			Ethalons[N] = 1;
			for (int q = 0; q < Iter; q++) {
				Inputs = Vectors[N];
				Hiddens = get_hiddens(Inputs, w12, T_Hid);
				Currents = get_result(Inputs, w12, T_Hid, w23, T_Out, Hiddens);
				for (int i = 0; i < NUM_OUT; i++)
					Mistakes[i] = Currents[i] - Ethalons[i];
				for (int j = 0; j < NUM_HID; j++) {
					for (int m = 0; m < NUM_OUT; m++) {
						MistakesHid[j] += Mistakes[m] * Currents[m] * (1 - Currents[m]) * w23[j][m];
					}
				}
				for (int j = 0; j < NUM_OUT; j++) {
					for (int i = 0; i < NUM_HID; i++) {
						w23[i][j] -= alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]) * Hiddens[i];
					}
					T_Out[j] += alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]);
				}
				for (int j = 0; j < NUM_HID; j++) {
					for (int i = 0; i < NUM_IN; i++) {
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

	bool Vectors3[] = { 1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	bool Vectors4[] = { 1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vectors5[] = { 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0 };
	bool Vectors6[] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0 };
	bool Vectors7[] = { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 };
	double* HiddenPred;
	double* Values;
	Vectors[3] = Vectors3;
	Vectors[4] = Vectors4;
	Vectors[5] = Vectors5;
	Vectors[6] = Vectors6;
	Vectors[7] = Vectors7;

	for (int i = 0; i < 8; i++) {
		Inputs = Vectors[i];
		cout << "Result of vector " << i + 1 << " which equals ";
		for (int j = 0; j < 20; j++) {
			cout << Inputs[j] << ' ';
		}
		cout << endl << "Result : ";
		HiddenPred = get_hiddens(Inputs, w12, T_Hid);
		Values = get_result(Inputs, w12, T_Hid, w23, T_Out, HiddenPred);
		cout << Values[0] << endl;
	}
	system("pause");
}
