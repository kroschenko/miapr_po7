#include <iostream>
#include <math.h>
#define NUMBER_IN 10
#define NUMBER_HID 4
#define NUMBER_OUT 1
#define e 2.71828
using namespace std;

double sigmoid(double x) {  //Сигмоид
	return 1 / (1 + pow(e, -x));
}

double* get_hiddens(bool* Inputs, double w12[NUMBER_IN][NUMBER_HID], double T_Hid[]) {
	double* Hiddens = new double[NUMBER_HID];
	for (int i = 0; i < NUMBER_HID; i++) Hiddens[i] = 0;
	for (int i = 0; i < NUMBER_HID; i++) {
		for (int j = 0; j < NUMBER_IN; j++) {
			Hiddens[i] += w12[j][i] * Inputs[j];
		}
		Hiddens[i] -= T_Hid[i];
		Hiddens[i] = sigmoid(Hiddens[i]);
	}
	return Hiddens;
}

double* get_result(bool* Inputs, double w12[NUMBER_IN][NUMBER_HID], double T_Hid[], double w23[NUMBER_HID][NUMBER_OUT], double T_Out[], double Hiddens[NUMBER_HID]) {
	double* Results = new double[NUMBER_OUT];
	for (int i = 0; i < NUMBER_OUT; i++)
		Results[i] = 0;
	for (int j = 0; j < NUMBER_OUT; j++) {
		for (int i = 0; i < NUMBER_HID; i++) {
			Results[j] += Hiddens[i] * w23[i][j];
		}
		Results[j] -= T_Out[j];
		Results[j] = sigmoid(Results[j]);
	}
	return Results;
}


int main() {
	///Vectora
	bool Vect1[] = { 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0 };
	bool Vect2[] = { 0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,0,0,0 };
	bool Vect3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 };
	///Input'bI i VectorbI
	bool* Inputs = new bool[NUMBER_IN];
	for (int i = 0; i < NUMBER_IN; i++) Inputs[i] = 0;
	bool** Vectors = new bool* [8];
	Vectors[0] = Vect1;
	Vectors[1] = Vect2;
	Vectors[2] = Vect3;
	///
	double w12[NUMBER_IN][NUMBER_HID], w23[NUMBER_HID][NUMBER_OUT], T_Hid[NUMBER_HID], T_Out[NUMBER_OUT], E_min = 0.0001, alpha = 0.04, Ethalon, E = 0, Outputs[NUMBER_OUT] = { 0 };
	double* Currents = new double[NUMBER_OUT];
	double* Hiddens = new double[NUMBER_HID];
	double Mistakes[NUMBER_OUT] = { 0 };
	double Ethalons[NUMBER_OUT] = { 0 };
	double MistakesHid[NUMBER_HID] = { 0 };
	int Iter = 1;  //KOLVO ITERACII  POKA 4to 1
	///Randomizacia
	for (int i = 0; i < NUMBER_IN; i++) {
		for (int j = 0; j < NUMBER_HID; j++) {
			w12[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < NUMBER_OUT; k++) {
				w23[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				T_Out[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			T_Hid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	int H = 0;
	do {
		E = 0;
		for (int N = 0; N < NUMBER_OUT; N++) {
			Ethalons[0] = 0;
			Ethalons[N] = 1;
			for (int q = 0; q < Iter; q++) {  //1 iteracia poka 4to
				///Result,Mistake
				Inputs = Vectors[N];
				Hiddens = get_hiddens(Inputs, w12, T_Hid);
				Currents = get_result(Inputs, w12, T_Hid, w23, T_Out, Hiddens);
				for (int i = 0; i < NUMBER_OUT; i++)
					Mistakes[i] = Currents[i] - Ethalons[i];
				///MistakesHid
				for (int j = 0; j < NUMBER_HID; j++) {
					for (int m = 0; m < NUMBER_OUT; m++) {
						MistakesHid[j] += Mistakes[m] * Currents[m] * (1 - Currents[m]) * w23[j][m];
					}
				}
				///Oby4enie 2->3
				for (int j = 0; j < NUMBER_OUT; j++) {
					for (int i = 0; i < NUMBER_HID; i++) {
						w23[i][j] -= alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]) * Hiddens[i];
					}
					T_Out[j] += alpha * Mistakes[j] * Currents[j] * (1 - Currents[j]);
				}
				///Oby4enie 1->2
				for (int j = 0; j < NUMBER_HID; j++) {
					for (int i = 0; i < NUMBER_IN; i++) {
						w12[i][j] -= alpha * MistakesHid[j] * Hiddens[j] * (1 - Hiddens[j]) * Inputs[i];
					}
					T_Hid[j] += alpha * MistakesHid[j] * Hiddens[j] * (1 - Hiddens[j]);
				}
				E += pow(Mistakes[N], 2);
			}
		}
		E /= 2;  //Probably
		if (H % 100 == 0 || H < 300) {
			cout << H << ";" << E << endl;
		}
		H++;

	} while (E > E_min);

	//Predictions:
	double* HiddenPred;
	double* Values;
	bool Vectors3[] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0 };
	bool Vectors4[] = { 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0 };
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
