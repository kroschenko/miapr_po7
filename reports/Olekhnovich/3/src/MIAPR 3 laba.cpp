#include <iostream>
using namespace std;

class InputNeuron {
public:
	double Wes;

	void random_Wes() {
		Wes = 2 * ((rand() % 10) * 0.1) - 1;
	}
	void izm_ves(double a, double y, double t, double x) {
		Wes -= a * (y - t) * x;
	}
	void set_Wes(int Wes) {
		this->Wes = Wes;
	}
	double get_Wes() {
		return Wes;
	}
};

class HiddenNeuron {
public:
	double Wes;
	double Sum, znach;

	void random_Wes() {
		Wes = 2 * ((rand() % 10) * 0.1) - 1;
	}
	void izm_ves(double a, double y1, double gamma, double y2) {
		Wes -= a * gamma * y1 * (1 - y1) * y2;
	}
};

int main() {
	setlocale(LC_ALL, "ru");
	system("color f0");

	const int number = 30,
		num_input_neuron = 8,
		num_hidden_neuron = 3;
	double a = 0.2, b = 0.2, c = 0.06, d = 0.2;
	double Em = 0.00001,
		A = 0.1,
		Emax;

	double standart_y[number + 15], y,
		Tx[num_hidden_neuron],
		Ty = 2 * ((rand() % 10) * 0.1) - 1,
		Error,
		Error_i[number - num_input_neuron];

	for (int i = 0; i < number; i++) {
		double x = 0.1 * i;
		standart_y[i] = a * cos(b * x) + c * sin(d * x);
	}

	InputNeuron W[num_hidden_neuron * num_input_neuron];
	HiddenNeuron Sig[num_hidden_neuron];

	for (int i = 0; i < num_input_neuron * num_hidden_neuron; i++) {
		W[i].random_Wes();
	}

	for (int i = 0; i < num_hidden_neuron; i++) {
		Sig[i].random_Wes();
		Sig[i].Sum = 0;
		Tx[i] = 2 * ((rand() % 10) * 0.1) - 1;
	}

	do {
		y = 0;
		Emax = 0;

		for (int Prohod = 0; Prohod < number - num_input_neuron; Prohod++) {
			y = 0;
			for (int i = 0; i < num_hidden_neuron; i++) {
				for (int j = 0; j < num_input_neuron; j++) {
					Sig[i].Sum += W[j].Wes * standart_y[Prohod + j];
				}
				Sig[i].Sum -= Tx[i];
				Sig[i].znach = (1 / (1 + exp(-Sig[i].Sum)));
				Sig[i].Sum = 0;
			}

			for (int i = 0; i < num_hidden_neuron; i++) {
				y += Sig[i].znach * Sig[i].Wes;
			}
			y -= Ty;

			Error = y - standart_y[Prohod + num_input_neuron];
			Error_i[Prohod] = Error;

			for (int i = 0; i < num_hidden_neuron; i++) {
				Sig[i].Wes -= A * Sig[i].znach * Error;
			}
			Ty += A * Error;

			for (int i = 0; i < num_hidden_neuron; i++) {
				for (int j = 0; j < num_input_neuron; j++)
					W[(i * 10) + j].Wes -= A * standart_y[Prohod + j] * (Sig[i].Wes * Error * Sig[i].znach * (1 - Sig[i].znach));
				Tx[i] += A * (Sig[i].Wes * Error * Sig[i].znach * (1 - Sig[i].znach));
			}
		}

		for (int i = 0; i < number - num_input_neuron; i++) {
			Emax += (pow(Error_i[i], 2.0) * 0.5);
		}
	} while (Emax > Em);

	cout << "Эталон" << "\t\t\t\t" << "Прогноз" << "\t\t\t\tОтклонение" << endl;

	for (int Proh = 0; Proh < 15; Proh++) {
		y = 0;
		for (int i = 0; i < num_hidden_neuron; i++) {
			for (int j = 0; j < num_input_neuron; j++) {
				Sig[i].Sum += W[j].Wes * standart_y[number - num_input_neuron + Proh + j];
			}
			Sig[i].Sum -= Tx[i];
			Sig[i].znach = (1 / (1 + exp(-Sig[i].Sum)));
			Sig[i].Sum = 0;
		}

		for (int i = 0; i < num_hidden_neuron; i++) {
			y += Sig[i].znach * Sig[i].Wes;
		}
		y -= Ty;

		standart_y[number + Proh] = y;
		double x = 0.1 * ((double)Proh + (double)number);
		cout << a * cos(b * x) + c * sin(d * x) << "\t\t\t";
		cout << standart_y[number + Proh] << "\t\t\t";
		cout << a * cos(b * x) + c * sin(d * x) - standart_y[number + Proh] << endl;
	}
	system("pause");
	return 0;
}
