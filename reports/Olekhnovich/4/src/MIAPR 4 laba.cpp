#include <iostream>
#include <cmath>

using namespace std;

double get_random(double LO = -1.0, double HI = 1.0);

int main()
{
    const int       random_seed = 1347652890;// Сюда пишем любое число, чтобы был разный рандом

    const double    a = 0.2;      // Параметр y = a * cos(b*x) + c * sin(d*x)
    const double    b = 0.2;      // Параметр y = a * cos(b*x) + c * sin(d*x)
    const double    c = 0.06;     // Параметр y = a * cos(b*x) + c * sin(d*x)
    const double    d = 0.2;      // Параметр y = a * cos(b*x) + c * sin(d*x)
    const double    step = 0.01;     // x = step * index
    double          x;
    double          y;

    const int       inputs = 8;        // Количество нейронов входного слоя
    const int       hiddens = 3;        // Количество нейронов скрытого слоя
    const int       outputs = 1;

    double          Ee = 1e-6;     // Желаемая средне квадратичная ошибка
    double          E;

    double          alpha_ki = 0.001;    // Скорость обучения от входного слоя до скрытого
    double          alpha_ij = 0.001;    // Скорость обучения от скрытого слоя до выходного
    double          alpha_sum1;
    double          alpha_sum2;
    double          alpha_sum3;
    double          alpha_x2;

    const int       number_learning = 30;       // Количество для обучения
    const int       number_test = 15;       // Количество для тестов

    double* e = new double[number_learning + number_test + inputs]; // Эталоны

    double** w_ki = new double* [inputs];     // Веса от входного до скрытого слоя
    for (int k = 0; k < inputs; ++k) w_ki[k] = new double[hiddens];

    double** w_ij = new double* [hiddens];    // Веса от скрытого до выходного слоя
    for (int i = 0; i < inputs; ++i) w_ij[i] = new double[outputs];

    double* T_i = new double[hiddens];     // Пороги скрытого слоя

    double* T_j = new double[outputs];     // Пороги скрытого слоя

    int             era;                                // Количество эпох
    int             q;
    int             k;                                  // Итератор для входного слоя
    int             i;                                  // Итератор для скрытого слоя
    int             j;                                  // Итератор для выходного слоя

    double* y_k = new double[hiddens];     // Значения на входном слое (эталоны)

    double* S_i = new double[hiddens];     // Взвешенная сумма для скрытого слоя
    double* y_i = new double[hiddens];     // Значение на скрытом слое (сигмоидная функция)

    double* S_j = new double[outputs];     // Взвешенная сумма для выходного слоя
    double* y_j = new double[outputs];     // Значение на скрытом слое (линейная функция)

    double* j_j = new double[outputs];     // Обратная ошибка выходного слоя
    double* j_i = new double[hiddens];     // Обратная ошибка скрытого слоя

    double          dF_i;                               // Переменная для вычисления обратной ошибки выходного слоя
    double          dF_j;                               // Переменная для вычисления обратной ошибки скрытого слоя
    // = = = = = = = = ~ ~ ~ ~ ~ ~ ~ ~ = = = = = = = = ~ ~ ~ ~ ~ ~ ~ ~ = = = = = = = =

    system("chcp 65001");
    srand(random_seed);
    printf(" Привет, мир! \n \n");


    // = = = = = Создаем эталонные значения = = = = =
    printf(" e = { \n");
    for (i = 0; i < number_learning + number_test + inputs; ++i)
    {
        x = step * i;
        y = a * cos(b * x) + c * sin(d * x);
        e[i] = y;
        printf(" %4d: %20.16f , \n", i + 1, e[i]);
    }
    printf(" } \n \n");


    // = = = = = Генерируем веса от входного до скрытого слоя = = = = =
    printf(" w_ki = { \n");
    for (k = 0; k < inputs; ++k)
    {
        printf(" %4d: { ", k + 1);
        for (i = 0; i < hiddens; ++i)
        {
            w_ki[k][i] = get_random();
            printf(" %4d: %20.16f ,", i + 1, w_ki[k][i]);
        }
        printf(" } , \n");
    }
    printf(" } \n \n");


    // = = = = = Генерируем веса от скрытого до выходного слоя = = = = =
    printf(" w_ij = { \n");
    for (i = 0; i < hiddens; ++i)
    {
        printf(" %4d: { ", i + 1);
        for (j = 0; j < outputs; ++j)
        {
            w_ij[i][j] = get_random();
            printf(" %4d: %20.16f ,", j + 1, w_ij[i][j]);
        }
        printf(" } , \n");
    }
    printf(" } \n \n");


    // = = = = = Генерируем пороги для скрытого слоя = = = = =
    printf(" T_i = { \n");
    for (i = 0; i < hiddens; ++i)
    {
        T_i[i] = get_random();
        printf(" %4d: %20.16f , \n", i + 1, T_i[i]);
    }
    printf(" } \n \n");


    // = = = = = Генерируем пороги для выходного слоя = = = = =
    printf(" T_j = { \n");
    for (j = 0; j < outputs; ++j)
    {
        T_j[j] = get_random();
        printf(" %4d: %20.16f , \n", j + 1, T_j[j]);
    }
    printf(" } \n \n");


    // = = = = = Правило Видроу-Хоффа = = = = =
    for (era = 1; ; ++era)
    {
        for (q = 0; q < number_learning - inputs; ++q)
        {
            // = = = = = Вычисляем значения на входном слое (взяли эталоны) = = = = =
//            printf (" y_k = { \n");
            for (k = 0; k < inputs; ++k)
            {
                y_k[k] = e[q + k];
                //                printf (" %4d: %20.16f , \n", k+1, y_k[k]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем взвешенную сумму скрытого слоя = = = = =
            //            printf (" S_i = { \n");
            for (i = 0; i < hiddens; ++i)
            {
                S_i[i] = 0;
                for (k = 0; k < inputs; ++k)
                {
                    S_i[i] += y_k[k] * w_ki[k][i];
                }
                //                printf (" %4d: %20.16f , \n", i+1, S_i[i]);
                S_i[i] -= T_i[i];
                //                printf (" %4d: %20.16f , \n", i+1, S_i[i]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем значения на скрытом слое (сигмоидная функция) = = = = =
            //            printf (" y_i = { \n");
            for (i = 0; i < hiddens; ++i)
            {
                y_i[i] = 1 / (1 + pow(-S_i[i], 2));
                //                printf (" %4d: %20.16f , \n", i+1, y_i[i]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем взвешенную сумму выходного слоя = = = = =
            //            printf (" S_j = { \n");
            for (j = 0; j < outputs; ++j)
            {
                S_j[j] = 0;
                for (i = 0; i < hiddens; ++i)
                {
                    S_j[j] += y_i[i] * w_ij[i][j];
                }
                //                printf (" %4d: %20.16f , \n", j+1, S_j[j]);
                S_j[j] -= T_j[j];
                //                printf (" %4d: %20.16f , \n", j+1, S_j[j]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем значения на выходном слое (линейная функция) = = = = =
            //            printf (" y_j = { \n");
            for (j = 0; j < outputs; ++j)
            {
                y_j[j] = S_j[j];
                //                printf (" %4d: %20.16f , \n", j+1, y_j[j]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем обратную ошибку для выходного слоя = = = = =
            //            printf (" j_j = { \n");
            for (j = 0; j < outputs; ++j)
            {
                j_j[j] = y_j[j] - e[q + inputs + j];
                //                printf (" %4d: %20.16f , \n", j+1, j_j[j]);
            }
            //            printf (" } \n \n");


                        // = = = = = Вычисляем обратную ошибку для скрытого слоя = = = = =
            dF_j = 1;
            //            printf (" j_i = { \n");
            for (i = 0; i < hiddens; ++i)
            {
                j_i[i] = 0;
                for (j = 0; j < outputs; ++j)
                {
                    j_i[i] += j_j[j] * dF_j * w_ij[i][j];
                }
                //                printf (" %4d: %20.16f , \n", i+1, j_i[i]);
            }
            //            printf (" } \n \n");


                        // 190 эпох - без alpha_ki
                        // 182 эпохи - с alpha_ki
                        // 92 эпохи - с alpha_ki * (-1)
                        // < < < < < Адаптивный шаг от входного до скрытого слоя (для сигмоидной функции)
            alpha_sum1 = 0;
            alpha_sum3 = 0;
            for (j = 0; j < outputs; ++j)
            {
                alpha_sum1 += pow(j_j[j], 2) * y_j[j] * (1 - y_j[j]);
                alpha_sum3 += pow(j_j[j], 2) * pow(y_j[j], 2) * pow(1 - y_j[j], 2);
            }
            //            printf(" alpha_sum1 = %20.16f \n", alpha_sum1);
            //            printf(" alpha_sum3 = %20.16f \n", alpha_sum3);

            alpha_sum2 = 0;
            for (i = 0; i < hiddens; ++i)
            {
                alpha_sum2 += pow(y_i[i], 2);
            }
            //            printf(" alpha_sum2 = %20.16f \n", alpha_sum2);

            alpha_ki = 4 * alpha_sum1 / ((1 + alpha_sum2) * alpha_sum3); // адаптивный шаг для сигмоидной функции
//            printf(" alpha_ki = %20.16f \n", alpha_ki);
            // > > > > >


            // < < < < < Адаптивнй шаг от скрытого до выходного слоя (для линейной функции)
            alpha_x2 = 0;
            for (i = 0; i < hiddens; ++i)
            {
                alpha_x2 += pow(e[q + i], 2);
            }
            alpha_ij = 1 / (1 + alpha_x2);
            // > > > > >


            // = = = = = Новые веса между  скрытым и выходным слоем = = = = =
//            printf (" w_ij = { \n");
            for (i = 0; i < hiddens; ++i)
            {
                //                printf (" %4d: { ", i+1);
                for (j = 0; j < outputs; ++j)
                {
                    w_ij[i][j] -= alpha_ij * j_j[j] * dF_j * y_j[j];
                    //                    printf (" %4d: %20.16f ,", j+1, w_ij[i][j]);
                }
                //                printf (" } , \n");
            }
            //            printf (" } \n \n");


                        // = = = = = Новые пороги для выходного слоя = = = = =
            //            printf (" T_j = { \n");
            for (j = 0; j < outputs; ++j)
            {
                T_j[j] += alpha_ij * j_j[j] * dF_j;
                //                printf (" %4d: %20.16f , \n", j+1, T_j[j]);
            }
            //            printf (" } \n \n");


                        // = = = = = Новые веса между входным и скрытым слоем = = = = =
            //            printf (" w_ki = { \n");
            for (k = 0; k < inputs; ++k)
            {
                //                printf (" %4d: { ", k+1);
                for (i = 0; i < hiddens; ++i)
                {
                    dF_i = y_i[i] * (1 - y_i[i]);
                    w_ki[k][i] -= alpha_ki * j_i[i] * dF_i * y_i[i];
                    //                    printf (" %4d: %20.16f ,", i+1, w_ki[k][i]);
                }
                //                printf (" } , \n");
            }
            //            printf (" } \n \n");


                        // = = = = = Новые пороги для скрытого слоя = = = = =
            //            printf (" T_i = { \n");
            for (i = 0; i < hiddens; ++i)
            {
                dF_i = y_i[i] * (1 - y_i[i]);
                T_i[i] += alpha_ki * j_i[i] * dF_i;
                //                printf (" %4d: %20.16f , \n", i+1, T_i[i]);
            }
            //            printf (" } \n \n");
            //            break;
        }
        //        break;

                // = = = = = Ошибка сети = = = = =
        E = 0;
        for (j = 0; j < outputs; ++j)
        {
            E += 0.5 * pow(y_j[j] - e[q + inputs + j], 2);
        }

        if (E < Ee)
        {
            printf(" era: %8d        Error: %20.16f < %-20.16f \n", era, E, Ee);
            break;
        }

        printf(" era: %8d        Error: %20.16f > %-20.16f \n", era, E, Ee);
    }
    printf(" \n \n");
    //    return 0;


        // = = = = = Веса от входного до скрытого слоя после обучения = = = = =
    printf(" w_ki = { \n");
    for (k = 0; k < inputs; ++k)
    {
        printf(" %4d: { ", k + 1);
        for (i = 0; i < hiddens; ++i)
        {
            printf(" %4d: %20.16f ,", i + 1, w_ki[k][i]);
        }
        printf(" } , \n");
    }
    printf(" } \n \n");


    // = = = = = Веса от скрытого до выходного слоя после обучения = = = = =
    printf(" w_ij = { \n");
    for (i = 0; i < hiddens; ++i)
    {
        printf(" %4d: { ", i + 1);
        for (j = 0; j < outputs; ++j)
        {
            printf(" %4d: %20.16f ,", j + 1, w_ij[i][j]);
        }
        printf(" } , \n");
    }
    printf(" } \n \n");


    // = = = = = Пороги для скрытого слоя после обучения = = = = =
    printf(" T_i = { \n");
    for (i = 0; i < hiddens; ++i)
    {
        printf(" %4d: %20.16f , \n", i + 1, T_i[i]);
    }
    printf(" } \n \n");


    // = = = = = Пороги для выходного слоя после обучения = = = = =
    printf(" T_j = { \n");
    for (j = 0; j < outputs; ++j)
    {
        printf(" %4d: %20.16f , \n", j + 1, T_j[j]);
    }
    printf(" } \n \n");


    // = = = = = Результаты после обучения = = = = =
    printf(" Результат после обучения на обученной выборке \n");
    printf(" %-8s %-20s %-20s %-20s %-20s \n",
        "#",
        "etalon",
        "prognoz",
        "otclonenie",
        "otclonenie^2"
    );
    for (q = 0; q < number_learning; ++q)
    {
        // = = = = = Вычисляем значения на входном слое (взяли эталоны) = = = = =
//        printf (" y_k = { \n");
        for (k = 0; k < inputs; ++k)
        {
            y_k[k] = e[q + k];
            //                printf (" %4d: %20.16f , \n", k+1, y_k[k]);
        }
        //            printf (" } \n \n");


                // = = = = = Вычисляем взвешенную сумму скрытого слоя = = = = =
        //        printf (" S_i = { \n");
        for (i = 0; i < hiddens; ++i)
        {
            S_i[i] = 0;
            for (k = 0; k < inputs; ++k)
            {
                S_i[i] += y_k[k] * w_ki[k][i];
            }
            //            printf (" %4d: %20.16f , \n", i+1, S_i[i]);
            S_i[i] -= T_i[i];
            //            printf (" %4d: %20.16f , \n", i+1, S_i[i]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем значения на скрытом слое (сигмоидная функция) = = = = =
        //        printf (" y_i = { \n");
        for (i = 0; i < hiddens; ++i)
        {
            y_i[i] = 1 / (1 + pow(-S_i[i], 2));
            //            printf (" %4d: %20.16f , \n", i+1, y_i[i]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем взвешенную сумму выходного слоя = = = = =
        //        printf (" S_j = { \n");
        for (j = 0; j < outputs; ++j)
        {
            S_j[j] = 0;
            for (i = 0; i < hiddens; ++i)
            {
                S_j[j] += y_i[i] * w_ij[i][j];
            }
            //            printf (" %4d: %20.16f , \n", j+1, S_j[j]);
            S_j[j] -= T_j[j];
            //            printf (" %4d: %20.16f , \n", j+1, S_j[j]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем значения на выходном слое (линейная функция) = = = = =
        //        printf (" y_j = { \n");
        for (j = 0; j < outputs; ++j)
        {
            y_j[j] = S_j[j];
            //            printf (" %4d: %20.16f , \n", j+1, y_j[j]);
        }
        //        printf (" } \n \n");

        printf(" %8d %20.16f %20.16f %20.16f %20.16f \n",
            q + 1,
            e[q + inputs],
            y_j[0],
            e[q + inputs] - y_j[0],
            pow(e[q + inputs] - y_j[0], 2)
        );
    }


    // = = = = = Результаты тестирования = = = = =
    printf(" Результат после обучения на неизвесной выборке (выборка не участвовала в обучении) \n");
    printf(" %-8s %-20s %-20s %-20s %-20s \n",
        "#",
        "etalon",
        "y_j",
        "etalon - y_j",
        "(etalon - y_j)^2"
    );
    for (q = 0; q < number_test; ++q)
    {
        // = = = = = Вычисляем значения на входном слое (взяли эталоны) = = = = =
//        printf (" y_k = { \n");
        for (k = 0; k < inputs; ++k)
        {
            y_k[k] = e[q + number_learning + k];
            //                printf (" %4d: %20.16f , \n", k+1, y_k[k]);
        }
        //            printf (" } \n \n");


                // = = = = = Вычисляем взвешенную сумму скрытого слоя = = = = =
        //        printf (" S_i = { \n");
        for (i = 0; i < hiddens; ++i)
        {
            S_i[i] = 0;
            for (k = 0; k < inputs; ++k)
            {
                S_i[i] += y_k[k] * w_ki[k][i];
            }
            //            printf (" %4d: %20.16f , \n", i+1, S_i[i]);
            S_i[i] -= T_i[i];
            //            printf (" %4d: %20.16f , \n", i+1, S_i[i]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем значения на скрытом слое (сигмоидная функция) = = = = =
        //        printf (" y_i = { \n");
        for (i = 0; i < hiddens; ++i)
        {
            y_i[i] = 1 / (1 + pow(-S_i[i], 2));
            //            printf (" %4d: %20.16f , \n", i+1, y_i[i]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем взвешенную сумму выходного слоя = = = = =
        //        printf (" S_j = { \n");
        for (j = 0; j < outputs; ++j)
        {
            S_j[j] = 0;
            for (i = 0; i < hiddens; ++i)
            {
                S_j[j] += y_i[i] * w_ij[i][j];
            }
            //            printf (" %4d: %20.16f , \n", j+1, S_j[j]);
            S_j[j] -= T_j[j];
            //            printf (" %4d: %20.16f , \n", j+1, S_j[j]);
        }
        //        printf (" } \n \n");


                // = = = = = Вычисляем значения на выходном слое (линейная функция) = = = = =
        //        printf (" y_j = { \n");
        for (j = 0; j < outputs; ++j)
        {
            y_j[j] = S_j[j];
            //            printf (" %4d: %20.16f , \n", j+1, y_j[j]);
        }
        //        printf (" } \n \n");

        printf(" %8d %20.16f %20.16f %20.16f %20.16f \n",
            q + number_learning + 1,
            e[q + number_learning],
            y_j[0],
            e[q + number_learning] - y_j[0],
            pow(e[q + number_learning] - y_j[0], 2)
        );
    }

    printf(" \n");
    printf(" %8s : %d \n", "seed", random_seed);
    printf(" %8s : %d \n", "era", era);
    printf(" %8s : %-20.16f \n", "y_j", y_j[0]);

    return (0);
}

double get_random(double LO, double HI)
{
    double          rand_number;
    // = = = = = = = = ~ ~ ~ ~ ~ ~ ~ ~ = = = = = = = = ~ ~ ~ ~ ~ ~ ~ ~ = = = = = = = =

    rand_number = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
    return (rand_number);
}
