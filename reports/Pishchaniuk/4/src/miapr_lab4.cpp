#include <iostream>
#include <math.h>
#include <iomanip>
using namespace std;

double Func(double x);
double Sigm_Func(double x);
double* Hidden(double x, double w1[2][6], double T[2]);
double Print(double x, double w1[2][6], double w2[2], double T[2 + 1]);
double Adapt_Step(double w2[], double err, double Print, double hiddens[]);

int main()
{
	setlocale(LC_ALL, "ru");

	double w1[2][6],
		w2[2],
		T[2 + 1],
		main,
		current,
		V = 0.4,
		newV = 0.4,
		x = 4,
		Emin = 0.00002,
		Emax = 0;

	int era = 0,
		max_era = 1200;

	for (int i = 0; i < 2; i++)
	{
		for (int k = 0; k < 6; k++)
		{
			w1[i][k] = ((double)rand() / RAND_MAX) * 0.005;
		}
		w2[i] = ((double)rand() / RAND_MAX) * 0.005;
		T[i] = ((double)rand() / RAND_MAX) * 0.005;
	}
	T[4] = ((double)rand() / RAND_MAX) * 0.005;
	do
	{
		Emax = 0;
		for (int q = 0; q < 300; q++)
		{
			current = Print(x, w1, w2, T);
			main = Func(x + 6 * 0.1);
			double err = current - main;
			double* hiddens = Hidden(x, w1, T);
			for (int j = 0; j < 2; j++)
				w2[j] -= V * err * hiddens[j];
			T[4] += V * err;
			for (int k = 0; k < 2; k++)
			{
				for (int i = 0; i < 6; i++)
					w1[k][i] -= newV * Func(x + i * 0.1) * hiddens[k] * (1 - hiddens[k]) * w2[k] * err;
				T[k] += newV * hiddens[k] * (1 - hiddens[k]) * w2[k] * err;
			}
			newV = Adapt_Step(w2, err, current, hiddens);
			x += 0.1;
			Emax += pow(err, 2);
		}
		Emax /= 2;
		era++;
	} while (Emax > Emin);
	cout << "Emax=  " << Emax << endl;
	cout << "Number of eras: " << era << endl;
	cout << setw(27) << left << "Reference values" << setw(29) << left << "The resulting values" << setw(30) << left << "Deviation" << endl;
	for (int i = 0; i < 100; i++)
	{
		double Resultat = Print(x, w1, w2, T), Etalon = Func(x + 6 * 0.1);
		cout << setw(27) << left << Etalon << setw(27) << left << Resultat << setw(30) << Resultat - Etalon << endl;
		x += 0.1;
	}
	system("pause");
	return 0;
}

double Func(double x) {
	return 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x);
}
double Sigm_Func(double x) {
	return 1 / (1 + pow(2, -x));
}
double* Hidden(double x, double w1[2][6], double T[2]) {
	double* fin = new double[2];
	for (int i = 0; i < 2; i++)
		fin[i] = 0;
	double input_neuron[6];
	for (int k = 0; k < 6; k++, x += 0.1)
		input_neuron[k] = Func(x);
	for (int i = 0; i < 2; i++)
	{
		for (int k = 0; k < 6; k++)
			fin[i] += input_neuron[k] * w1[i][k];
		fin[i] -= T[i];
		fin[i] = Sigm_Func(fin[i]);
	}
	return fin;
}
double Print(double x, double w1[2][6], double w2[2], double T[2 + 1])
{
	double Resultat = 0;
	double* hidden_neuron = Hidden(x, w1, T);
	for (int j = 0; j < 2; j++) {
		Resultat += hidden_neuron[j] * w2[j];
	}
	Resultat -= T[4];
	return Resultat;
}
double Adapt_Step(double w2[], double err, double Print, double hiddens[])
{
	double V = 0, e1 = 0, e2 = 0;
	for (int i = 0; i < 2; i++)
	{
		e1 += pow(err * w2[i] * (1 - hiddens[i]) * hiddens[i], 2) * hiddens[i] * (1 - hiddens[i]);
		e2 += pow(err * w2[i] * (1 - hiddens[i]) * hiddens[i], 2) * hiddens[i] * hiddens[i] * (1 - hiddens[i]) * (1 - hiddens[i]);
	}
	V = 4 * e1 / (e2 * (1 + Print * Print));
	return V;
}
