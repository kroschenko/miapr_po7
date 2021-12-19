from math import sin
from random import uniform


def function(s):
    return 3 * sin(5 * s) + 0.5


def y_calculating(w, x, T):
    s = 0
    for j in range(4):
        s += w[j] * x[j]

    return s - T


alpha = 0.05  # Скорость обучения
min_error = 1e-28  # Желанная минимальная ошибка

xe_train = [function(i / 10) for i in range(34)]  # Эталонные значения для обучения
xe_test = [function(i / 10) for i in range(30, 49)]  # Эталонные значения для тестирования

w = [uniform(0, 1) for _ in range(4)]  # Создание случайных весов
T = uniform(0, 1)  # Создание случайного порога

repeat = 0  # Счетчик повторений
error = 1  # Среднеквадратичная ошибка = 1, чтобы пройти первый while

# Обучение
while error >= min_error and repeat < 100:  # Остановка по достижению мин. ошибки или 100-ого повтора
    error = 0
    repeat += 1
    for epoch in range(30):  # 30 выборок
        y = y_calculating(w, xe_train[epoch:epoch + 4], T)  # Вычисление (y) нейронной сетью
        e_out: float = xe_train[epoch + 4]  # Эталонное значение этой выборки
        delta: float = y - e_out  # Разница
        error += delta ** 2 / 2  # Подсчет среднеквадратичной ошибки

        alpha = 1 / (1 + sum([x ** 2 for x in xe_train[epoch:epoch + 4]]))  # Адаптивная скорость

        for t in range(4):  # Изменение всех весов и порога
            w[t] -= alpha * delta * xe_train[epoch + t]
        T += alpha * delta

# Конец обучения и начало тестирования

print(f'Нейросеть обучена за {repeat} повтор.')
print('Тестирование:')
print(' N:     Идеальное значение    Полученное значение                   Разница          Среднекв. ошибка')

for epoch in range(15):
    y = y_calculating(w, xe_test[epoch:epoch + 4], T)  # Вычисление (y) нейронной сетью

    e_out: float = xe_test[epoch + 4]  # Эталонное значение
    delta: float = y - e_out  # Разница
    error += delta ** 2 / 2  # Подсчет среднеквадратичной ошибки

    print(f'{epoch + 1:2}:  {e_out:21}  {y:21}  {delta:24}  {error:24}')
