from random import uniform, randint
from math import exp

vector_1 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector_2 =  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
vector_3 =  [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
vectors = [vector_1, vector_2, vector_3]

omega = [[[uniform(0, 1) for _ in range(6)] for _ in range(2)], [uniform(0,1) for _ in range(2)]] # коэффициенты
T = [0 for _ in range(3)] # пороги обучения для двух скрытых нейронов и одного выходного
count_of_epochs = 1 # порядковый номер эпох из 3 итераций
last_err = 0 # предыдущая погрешность относительно текущей

Err_min = float(input('Введите максимально возможную квадратичную погрешность при обучении: '))
Err_sum = 1 # текущая ошибка

print('Даны следующие вектора:')
print('Вектор 1:', vector_1, '\nВектор 2:', vector_2, '\nВектор 3:', vector_3)

while Err_sum >= Err_min:
  Err_sum = 0 # для высчитывания тестовой среднеквадратической ошибки
  for vector_number in range(3):
    sum_sq_y = 0
    x = vectors[vector_number] # для обучения, проверять на практике будем другие значения
    y_prom_test, S_prom_test, gamma_prom_test = [0, 0], [0, 0], [0, 0]
    # выходные значения, S и погрешности для промежуточного слоя (тестовые)
    t = 1 # идеальное выходное значение
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
  if S3 == 0:
    break
  alpha_1 = S1 / S2 / S3
  alpha_2 = 1 / (1 + sum_sq_y)
  for j in range(2):
    for k in range(6):
      omega[0][j][k] -= alpha_1 * gamma_prom_test[j] * y_prom_test[j] * (1 - y_prom_test[j]) * x[k]
    T[j] += alpha_1 * gamma_prom_test[j] *  y_prom_test[j] * (1 - y_prom_test[j])
    omega[1][j] -= alpha_2 * delta * y_prom_test[j]
  T[2] += alpha_2 * delta
  avg_Err = Err_sum / 3
  if avg_Err != last_err:
    if avg_Err < Err_min:
      break
  last_err = avg_Err
  count_of_epochs += 1

print('\nПосле обучения на', count_of_epochs, 'эпохах имеем следующие конфигурации:\nКоэффициенты -', omega, '\nПороги функции T -', T)
print('\nПроводим тестирование/предсказание для вектора для кодового расстояния от 0 до 20 (каждое - по 1000 раз)\n')
error_sum = 0 # для высчитывания итоговой среднеквадратической ошибки
vector_difference = 0 # кодовое расстояние, для кторого будет проверяться НС
while vector_difference <= 20:
    test_number = 1 # номер теста
    error_mid_sum = 0 # для вычисления среднеквадратической ошибки при некотором расстоянии
    while test_number <= 1000:
      x = vectors[randint(0, 2)] # значения x для генерации ответа
      indexes_to_change = [] # значения индексов, которые нужно инвертировать
      while len(indexes_to_change) < vector_difference:
        cand_index = randint(0, 19)
        if cand_index not in indexes_to_change:
            indexes_to_change.append(cand_index)
      for index in indexes_to_change:
        x[index] = (x[index] + 1) % 2 # инвертация
      y_prom, S_prom, gamma_prom = [0, 0], [0, 0], [0, 0]
      # выходные значения, S и погрешности для промежуточного слоя (при прогнозировании)
      ideal = 1 # идеальное выходное значение
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
      error_mid_sum += err
      test_number += 1
    avg_mid_error = error_mid_sum / 15
    print('Среднеквадратическая погрешность для кодового расстояния', vector_difference, 'равна', avg_mid_error)
    error_sum += avg_mid_error
    vector_difference += 1

avg_error = error_sum / 20 # среднеквадратическая погрешность
print('\nИтоговая среднеквадратическая погрешность равна', avg_error)
