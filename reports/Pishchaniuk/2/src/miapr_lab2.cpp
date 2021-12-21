#include <iostream>
#include <iomanip>
#include <Windows.h>
using namespace std;

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    int a = 4,
        b = 8,
        inputs_number = 3, 
        n_obychenie = 30, 
        n_prognoz = 15, 
        era = 0;

    double d = 0.4,
        x = 0,
        h = 0.1, 
        min_error = 0.001, 
        sum_error,
        T = 1; 
    double* w = new double[inputs_number]; 
    for (int i = 0; i < inputs_number; i++) { 
        w[i] = rand() % 100 * 0.1;
    }

    double* main = new double[n_obychenie + n_prognoz]; 
    for (int i = 0; i < n_obychenie + n_prognoz; i++) { 
        x += h;
        main[i] = a * sin(b * x) + d;
    }

    double y, 
        V = 0.0003; 
    do {
        sum_error = 0;
        for (int i = 0; i < n_obychenie - inputs_number; i++) {
            y = 0;
            for (int j = 0; j < inputs_number; j++) {
                y += w[j] * main[i + j];
            }
            y -= T;
            for (int j = 0; j < inputs_number; j++) {
                w[j] -= V * (y - main[i + inputs_number]) * main[i + j];
            }
            sum_error += 0.5 * pow((y - main[i + inputs_number]), 2);
            T += V * (y - main[i + inputs_number]);

            double t = 0.0;
            for (int j = 0; j < inputs_number; j++) {
                t += pow(main[i + j], 2);
            }
            V = 1 / (1 + t); 
        }
        era++;
    } while (sum_error > min_error);

    cout << "Number of eras: " << era << endl;
    cout << "Learning Outcomes" << endl;
    cout << setw(27) << left << "Reference values" << setw(29) << left << "The resulting values" << setw(30) << left << "Deviation" << endl;

    double* prognoz = new double[n_obychenie + n_prognoz];
    for (int i = 0; i < n_obychenie; i++) { 
        prognoz[i] = 0;
        for (int j = 0; j < inputs_number; j++) {
            prognoz[i] += w[j] * main[i + j];
        }
        prognoz[i] -= T;

        cout << "y[" << i << "] = " << setw(25) << left << main[i + inputs_number] << setw(25) << left;
        cout << prognoz[i] << setw(30) << left << pow(main[i + inputs_number] - prognoz[i], 2) << endl;
    }

    cout << "Learning Outcomes" << endl;
    cout << setw(27) << left << "Reference values" << setw(29) << left << "The resulting values" << setw(30) << left << "Deviation" << endl;

    for (int i = 0; i < n_prognoz; i++) { 
        prognoz[i + n_obychenie] = 0;
        for (int j = 0; j < inputs_number; j++) {
            prognoz[i + n_obychenie] += w[j] * main[n_obychenie + j + i - inputs_number];
        }
        prognoz[i + n_obychenie] -= T;

        cout << "y[" << n_obychenie + i << "] = " << setw(25) << left << main[i + n_obychenie] << setw(25) << left;
        cout << prognoz[i + n_obychenie] << setw(30) << left << pow(main[i + n_obychenie] - prognoz[i + n_obychenie], 2) << endl;
    }
    delete[] w;
    delete[]main;
    delete[]prognoz;
    system("pause");
    return 0;
}
