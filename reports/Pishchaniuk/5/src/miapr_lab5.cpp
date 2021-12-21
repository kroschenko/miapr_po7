#include <iostream>
using namespace std;

double func(double x) {
	return 1 / (1 + pow(2.7, -x));
}

double* Hiddens(bool* inputs, double w12[20][40], double T_hid[]) {
	double* hiddens = new double[40];
	for (int i = 0; i < 40; i++) hiddens[i] = 0;
	for (int i = 0; i < 40; i++) {
		for (int j = 0; j < 20; j++) {
			hiddens[i] += w12[j][i] * inputs[j];
		}
		hiddens[i] -= T_hid[i];
		hiddens[i] = func(hiddens[i]);
	}
	return hiddens;
}

double* fin(bool* Inputs, double w12[20][40], double T_hid[], double w23[40][3], double T_out[], double hiddens[40]) {
	double* result = new double[3];
	for (int i = 0; i < 3; i++)
		result[i] = 0;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 40; i++) {
			result[j] += hiddens[i] * w23[i][j];
		}
		result[j] -= T_out[j];
		result[j] = func(result[j]);
	}
	return result;
}

int main() {
	setlocale(LC_ALL, "ru");
	int era = 0;
	bool Vector1[] = { 1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 }; 
	bool Vector2[] = { 1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 }; 
	bool Vector3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 }; 

	bool* inputs = new bool[20];
	for (int i = 0; i < 20; i++) inputs[i] = 0;
	bool** vectors = new bool* [12];
	vectors[0] = Vector1;
	vectors[1] = Vector2;
	vectors[2] = Vector3;

	double w12[20][40], w23[40][3], T_hid[40], T_out[3], E_min = 0.001, V = 0.04, Emax = 0, outputs[3] = { 0 };
	double* currents = new double[3];
	double* hiddens = new double[40];
	double errors[3] = { 0 };
	double mains[3] = { 0 };
	double error_hid[40] = { 0 };

	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 40; j++) {
			w12[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < 3; k++) {
				w23[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				T_out[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			T_hid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	do {
		Emax = 0;
		for (int N = 0; N < 3; N++) {
			mains[0] = 0;
			mains[N] = 1;
			inputs = vectors[N];
			hiddens = Hiddens(inputs, w12, T_hid);
			currents = fin(inputs, w12, T_hid, w23, T_out, hiddens);
			for (int i = 0; i < 3; i++)
				errors[i] = currents[i] - mains[i];
			for (int j = 0; j < 40; j++) {
				for (int m = 0; m < 3; m++) {
					error_hid[j] += errors[m] * currents[m] * (1 - currents[m]) * w23[j][m];
				}
			}
			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < 40; i++) {
					w23[i][j] -= V * errors[j] * currents[j] * (1 - currents[j]) * hiddens[i];
				}
				T_out[j] += V * errors[j] * currents[j] * (1 - currents[j]);
			}
			for (int j = 0; j < 40; j++) {
				for (int i = 0; i < 20; i++) {
					w12[i][j] -= V * error_hid[j] * hiddens[j] * (1 - hiddens[j]) * inputs[i];
				}
				T_hid[j] += V * error_hid[j] * hiddens[j] * (1 - hiddens[j]);
			}
			Emax += pow(errors[N], 2);
		}
		Emax /= 2;
		era++;
	} while (Emax > E_min);

	cout << "Number of eras: " << era << endl;
	cout << endl;
	double* hidden_pred;
	double* values;
	bool Vector11[] = { 1,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 }; 
	bool Vector12[] = { 1,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	bool Vector13[] = { 1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	bool Vector21[] = { 1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 }; 
	bool Vector22[] = { 1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vector23[] = { 1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vector31[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1 }; 
	bool Vector32[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,1,0,1,1 };
	bool Vector33[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,0,0,1,1 };

	vectors[3] = Vector11;
	vectors[4] = Vector12;
	vectors[5] = Vector13;
	vectors[6] = Vector21;
	vectors[7] = Vector22;
	vectors[8] = Vector23;
	vectors[9] = Vector31;
	vectors[10] = Vector32;
	vectors[11] = Vector33;

	for (int i = 0; i < 12; i++) {
		inputs = vectors[i];
		cout << "Vector " << i + 1 << ": ";
		for (int j = 0; j < 20; j++) {
			cout << inputs[j] << ' ';
		}
		cout << endl << "Result: ";
		hidden_pred = Hiddens(inputs, w12, T_hid);
		values = fin(inputs, w12, T_hid, w23, T_out, hidden_pred);
		cout << values[0] << ' ' << values[1] << ' ' << values[2] << endl;
		cout << endl;
	}
	system("pause");
	return 0;
}
