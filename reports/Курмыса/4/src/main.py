from random import uniform
from math import sin, cos, exp

def f(x):
  return 0.2 * cos(0.4 * x) + 0.09 * sin(0.4 * x)

omega = [[[uniform(0, 1) for _ in range(6)] for _ in range(2)], [uniform(0,1) for _ in range(2)]] # коэффициенты
T = [0 for _ in range(3)] # пороги обучения для двух скрытых нейронов и одного выходного
count_of_iterations = 1 # порядковый номер итерации
last_err = 0 # предыдущая погрешность относительно текущей

Err_min = float(input('Введите максимально возможную квадратичную погрешность при обучении: '))

print('Изначальные данные:\nИзначальные коэфф. -', omega, '\nПороги функций T -', T)
# обучение на тридцати итерациях, после которых мы получаем коэффициенты и T для непосредственного прогноза

print(f'Обучение:\nЭпоха #    N:  {"Идеальное значение t":21} {"Полученное значение":21} {"Отклонение между знач.":26} {"Ср. квадр. погр.":26}')
while True:
  Err_sum = 0 # для высчитывания тестовой среднеквадратической ошибки
  for i in range(30):
    sum_sq_y = 0
    x = [f(j / 10) for j in range(i, i + 6)] # для обучения, проверять на практике будем другие значения
    X = (i + 6) / 10 # аргумент функции для проверки погрешности
    y_prom_test, S_prom_test, gamma_prom_test = [0, 0], [0, 0], [0, 0]
    # выходные значения, S и погрешности для промежуточного слоя (тестовые)
    t = f(X) # идеальное значение f(X) напрямую
    for j in range(2):
      for k in range(6):
        S_prom_test[j] += omega[0][j][k] * x[k]
      S_prom_test[j] -= T[j]
      y_prom_test[j] = 1 / (1 + exp(-S_prom_test[j]))
      sum_sq_y += y_prom_test[j]**2
    S = 0 # функция активации для выходного нейрона (линейная :3)
    for j in range(2):
      S += omega[1][j] * y_prom_test[j]
    y = S - T[2] # предполагаемое выходное значение
    delta = y - t # погрешность между идеальным и полученным значениями
    err = delta**2 / 2
    Err_sum += err
    for j in range(2):
      gamma_prom_test[j] = delta * y * (1 - y) * omega[1][j]
    S1, S2, S3 = 0, 1, 0
    for j in range(2):
      S1 += 4 * gamma_prom_test[j]**2 * y_prom_test[j] * (1 - y_prom_test[j])
      S2 += y**2
      S3 += gamma_prom_test[j]**2 * y_prom_test[j]**2 * (1 - y_prom_test[j])**2
    alpha_1 = S1 / S2 / S3
    alpha_2 = 1 / (1 + sum_sq_y)
    for j in range(2):
      for k in range(6):
        omega[0][j][k] -= alpha_1 * gamma_prom_test[j] * y_prom_test[j] * (1 - y_prom_test[j]) * x[k]
      T[j] += alpha_1 * gamma_prom_test[j] *  y_prom_test[j] * (1 - y_prom_test[j])
      omega[1][j] -= alpha_2 * delta * y_prom_test[j]
    T[2] += alpha_2 * delta
  avg_Err = Err_sum / 30
  if avg_Err != last_err:
    if count_of_iterations % 500 == 0:
      print(f'Эпоха # {count_of_iterations:4}: {t:21} {y:21} {err:24} {avg_Err:24}')
    if avg_Err < Err_min:
      break
  last_err = avg_Err
  count_of_iterations += 1

print('\nПосле обучения на', count_of_iterations, 'итерациях(и, й) имеем следующие конфигурации:\nКоэффициенты -', omega, '\nПороги функции T -', T)
print('\nПроводим тестирование/предсказание значения функции на 15-ти случайных рациональных значениях от -100 до 100:')
test_number = 1 # номер теста
error_sum = 0 # для высчитывания итоговой среднеквадратической ошибки
print(f'Тестирование:\nЭпоха # N: {"Идеальное значение t":21} {"Полученное значение":21} {"Отклонение между знач.":26} {"Квадр. погр.":26}')
while test_number <= 15:
  X = uniform(-100, 100) # аргумент для тестирования
  x = [f(X - (i / 10)) for i in range(6, 0, -1)] # значения x для генерации ответа
  y_prom, S_prom, gamma_prom = [0, 0], [0, 0], [0, 0]
  # выходные значения, S и погрешности для промежуточного слоя (при прогнозировании)
  ideal = f(X) # идеальное значение функции
  for j in range(2):
      for k in range(6):
        S_prom[j] += omega[0][j][k] * x[k]
      S_prom[j] -= T[j]
      y_prom[j] = 1 / (1 + exp(-S_prom[j]))
  S = 0 # сумматор произведений
  for j in range(2):
    S += omega[1][j] * y_prom[j]
  y = S - T[2] # предполагаемое выходное значение
  delta = y - ideal
  err = delta**2 / 2
  error_sum += err
  print(f'Эпоха #{test_number:2}: {ideal:21} {y:21} {delta:24} {err:24}')
  test_number += 1

avg_error = error_sum / 15 # среднеквадратическая погрешность
print('\nСреднеквадратическая погрешность равна', avg_error)