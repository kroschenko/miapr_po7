import numpy as np

def calc_S_hid(S_hid):
    return 1 / (1 + np.exp(-S_hid))

def training(inputs, ideal_outputs, w_out, w_hid, T_out, T_hid, epochs = 1000):
    for epoch in range(epochs + 1):
        for r_i in np.random.choice(3, 3, replace=False):
            change_parametres(inputs[r_i], ideal_outputs[r_i], w_out, w_hid, T_out, T_hid)
    return w_out, w_hid, T_out, T_hid

def change_parametres(inputs, outputs, w_out, w_hid, T_out, T_hid):
    alpha = 0.1
    y_hid = calc_S_hid(np.dot(w_hid, inputs) - T_hid)
    y = calc_S_hid(np.dot(w_out, y_hid) - T_out)
    gamma_out = (y - outputs).reshape(1, -1)
    gamma_hid = (gamma_out * w_out.T * y * (1 - y)).sum(axis=1).reshape(1, -1)
    w_out -= alpha * gamma_out.reshape(-1, 1) * y_hid * (y * (1 - y)).reshape(-1, 1)
    T_out += (alpha * gamma_out * y * (1 - y)).reshape(-1)
    w_hid -= alpha * gamma_hid.reshape(-1, 1) * inputs.reshape(1, -1) * (y_hid * (1 - y_hid)).reshape(-1, 1)
    T_hid += alpha * gamma_hid.reshape(-1) * y_hid * (1 - y_hid)

def predict(inputs, w_hid, w_out, T_hid, T_out):
    hidden_outputs = calc_S_hid(np.dot(w_hid, inputs) - T_hid)
    return calc_S_hid(np.dot(w_out, hidden_outputs) - T_out)

def reverse_bit(arr, index):
    arr[index] ^= 1
    return arr

v_predict = np.vectorize(predict, excluded=[1, 2, 3, 4])

def main():
    input = 20
    hidden = 2
    out = 3
    input_bits = np.array([
        [0, 1, 0, 0, 1,	1, 0, 1, 0,	0, 0, 0, 1,	0, 1, 0, 1,	0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    ])
    ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    epoch = 30000
    w_hid = np.random.normal(0, 1, (hidden, input))
    w_out = np.random.normal(0, 1, (out, hidden))
    T_hid = np.random.normal(0, 1, hidden)
    T_out = np.random.normal(0, 1, out)
    w_out, w_hid, T_out, T_hid = training(
input_bits, ref, w_out.copy(), w_hid.copy(), T_out.copy(), T_hid.copy(), epoch)
    print('True означает что NN распознала вектор, False - нет')
    for i, ideal_out in enumerate(ref):
        print(f"{i + 1} Вектор:")
        cur_vector = input_bits[i]
        output = predict(cur_vector, w_hid, w_out, T_hid, T_out)
        is_recognize = ideal_out[output.argmax()] == 1
        print(f"Инвертировано 0 бит: {is_recognize}")
        inverted_arr = cur_vector.copy()
        for j_i, j in enumerate(np.random.choice(20, 20, replace=False)):
            reverse_bit(inverted_arr, j)
            output = predict(inverted_arr, w_hid, w_out, T_hid, T_out)
            is_recognize = ideal_out[output.argmax()] == 1
            print(f"Инвертировано {j_i + 1} бит: {is_recognize}")
main()
