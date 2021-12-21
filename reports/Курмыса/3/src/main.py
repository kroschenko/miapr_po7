from random import uniform
from math import sin, cos, exp

def f(x):
  return 0.2 * cos(0.4 * x) + 0.09 * sin(0.4 * x)

omega = [[[uniform(0, 1) for _ in range(6)] for _ in range(2)], [uniform(0,1) for _ in range(2)]] # ������������
T = [0 for _ in range(3)] # ������ �������� ��� ���� ������� �������� � ������ ���������
count_of_iterations = 1 # ���������� ����� ��������
last_err = 0 # ���������� ����������� ������������ �������

Err_min = float(input('������� ����������� ��������� ������������ ����������� ��� ��������: '))
alpha = float(input('������� ��� �������� alpha (0 < alpha < 1): '))

print('����������� ������:\n����������� �����. -', omega, '\n������ ������� T -', T)
# �������� �� �������� ���������, ����� ������� �� �������� ������������ � T ��� ����������������� ��������

print(f'��������:\n����� #  N:   {"��������� �������� t":21} {"���������� ��������":21} {"���������� ����� ����.":26} {"��. �����. ����.":26}')
while True:
  Err_sum = 0 # ��� ������������ �������� �������������������� ������
  for i in range(30):
    x = [f(j / 10) for j in range(i, i + 6)] # ��� ��������, ��������� �� �������� ����� ������ ��������
    X = (i + 6) / 10 # �������� ������� ��� �������� �����������
    y_prom_test, S_prom_test, gamma_prom_test = [0, 0], [0, 0], [0, 0]
    # �������� ��������, S � ����������� ��� �������������� ���� (��������)
    t = f(X) # ��������� �������� f(X) ��������
    for j in range(2):
      for k in range(6):
        S_prom_test[j] += omega[0][j][k] * x[k]
      S_prom_test[j] -= T[j]
      y_prom_test[j] = 1 / (1 + exp(-S_prom_test[j]))
    S = 0 # ������� ��������� ��� ��������� ������� (�������� :3)
    for j in range(2):
      S += omega[1][j] * y_prom_test[j]
    y = S - T[2] # �������������� �������� ��������
    delta = y - t # ����������� ����� ��������� � ���������� ����������
    err = delta**2 / 2
    Err_sum += err
    for j in range(2):
      gamma_prom_test[j] = delta * y * (1 - y) * omega[1][j]
      for k in range(6):
        omega[0][j][k] -= alpha * gamma_prom_test[j] * y_prom_test[j] * (1 - y_prom_test[j]) * x[k]
      T[j] += alpha * gamma_prom_test[j] *  y_prom_test[j] * (1 - y_prom_test[j])
      omega[1][j] -= alpha * delta * y_prom_test[j]
    T[2] += alpha * delta
  avg_Err = Err_sum / 30
  if avg_Err != last_err:
    if count_of_iterations % 1000 == 0:
      print(f'����� #{count_of_iterations // 1000:2}K: {t:21} {y:21} {err:24} {avg_Err:24}')
    if avg_Err < Err_min:
      break
  last_err = avg_Err
  count_of_iterations += 1

print('\n����� �������� ����� ��������� ������������:\n������������ -', omega, '\n������ ������� T -', T)
print('\n�������� ������������/������������ �������� ������� �� 15-�� ��������� ������������ ��������� �� -100 �� 100:')
test_number = 1 # ����� �����
error_sum = 0 # ��� ������������ �������� �������������������� ������
print(f'������������:\n����� # N: {"��������� �������� t":21} {"���������� ��������":21} {"���������� ����� ����.":26} {"�����. ����.":26}')
while test_number <= 15:
  X = uniform(-100, 100) # �������� ��� ������������
  x = [f(X - (i / 10)) for i in range(6, 0, -1)] # �������� x ��� ��������� ������
  y_prom, S_prom, gamma_prom = [0, 0], [0, 0], [0, 0]
  # �������� ��������, S � ����������� ��� �������������� ���� (��� ���������������)
  ideal = f(X) # ��������� �������� �������
  for j in range(2):
      for k in range(6):
        S_prom[j] += omega[0][j][k] * x[k]
      S_prom[j] -= T[j]
      y_prom[j] = 1 / (1 + exp(-S_prom[j]))
  S = 0 # �������� ������������
  for j in range(2):
    S += omega[1][j] * y_prom[j]
  y = S - T[2] # �������������� �������� ��������
  delta = y - ideal
  err = delta**2 / 2
  error_sum += err
  print(f'����� #{test_number:2}: {ideal:21} {y:21} {delta:24} {err:24}')
  test_number += 1

avg_error = error_sum / 15 # �������������������� �����������
print('\n�������������������� ����������� �����', avg_error)