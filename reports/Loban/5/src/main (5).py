import numpy as np


class NN:
    """Класс нейронки"""

    def __init__(self):
        """Нейронная сеть с количеством нейронов: 8-3-1"""
        self.wx = np.random.normal(-0.5, 0.5, (20, 3))
        self.th = np.random.normal(-0.5, 0.5, 3)
        self.wh = np.random.normal(-0.5, 0.5, (3, 3))
        self.ty = np.random.normal(-0.5, 0.5, 3)
        # Задание весов и порогов нормальным распределением нужных длин

    def go(self, x):
        """Прохождение всей нейронки"""
        self.x = x  # Входные параметры (1, 20)
        self.sh = np.dot(self.x, self.wx) - self.th  # Вектор сумм ((1, 20) x (20, 3) - (1, 3) = (1, 3))
        self.h = 1.0 / (1.0 + np.exp(-self.sh))  # Активация (1, 3)
        self.sy = np.dot(self.h, self.wh) - self.ty  # Вектор сумм ((1, 3) x (3, 3) - (1, 3) = (1, 3))
        self.y = self.sy  # Активация и получение выхода нейронки (1, 3)
        return self.y  # (1, 3)

    def back_propagation(self, error_y, alpha):
        """Обратное распространение ошибки с изменением весов, порога"""
        error_h = np.dot(error_y, self.wh.transpose())
        # Ошибка для скрытого слоя ((1, 3) x (3, 3).T = (1, 3))

        gamma_y = alpha * error_y
        self.wh -= np.dot(self.h.reshape(-1, 1), gamma_y.reshape(1, -1))
        self.ty += gamma_y

        gamma_h = alpha * error_h * self.y * (1.0 - self.y)
        self.wx -= np.dot(self.x.reshape(-1, 1), gamma_h.reshape(1, -1))
        self.th += gamma_h

    def learn(self, x, e, alpha):
        """Обучение наборами"""
        square_error = 0  # Среднеквадратичная ошибка
        for i in range(len(e)):
            y = self.go(x[i])  # Прогон выборки
            error = y - e[i]  # Ошибка нейронной сети
            square_error += np.sum(error ** 2 / 2)  # Суммирование среднеквадратичной ошибки
            self.back_propagation(error, alpha)  # Обратное распространение
        return square_error


def main():
    vectors = np.array(
        [
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        ]
    )
    e = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )

    nn = NN()
    alpha = 0.06

    for i in range(30_000):
        error = nn.learn(vectors, e, alpha)
        # print(f"Square error {i: 5}: {error: .8f}")  # Обучение с выводом ошибки
        if error <= 1e-3:
            break

    print(f"Отклонение %  0  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100", end="")
    for i in range(3):
        print(f"\nВектор   #{i}:", end="")
        y = nn.go(vectors[i])  # Проверка без отклонений
        result = y.argmax() == i  # Результатом считается тот выходной нейрон, где большее значение
        print(f"{result: 3}", end="")

        for j in np.random.choice(20, 20, replace=False):  # Случайный j до 20 без повторений
            vectors[i][j] ^= 1  # Смена бита
            y = nn.go(vectors[i])
            result = y.argmax() == i
            print(f"{result: 3}", end="")


if __name__ == '__main__':
    main()
