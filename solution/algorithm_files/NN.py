import numpy as np
import data_preprocessing as dp
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# X_train, y_train, X_test, y_test = dp.dataPreprocessing('D:\\Collage\\4th_year\\second_semester\\Data Mining\\Assignment(3)\\Kidney_Disease_data_for_Classification_V2.csv', 70)

def NN(X_train, y_train, X_test, y_test):
    np.random.seed(42)

    input_size = 24
    hidden_size = 4
    output_size = 1

    W = []
    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    W.append(W1)
    W.append(W2)

    B = []
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, output_size))
    B.append(b1)
    B.append(b2)

    learning_rate = 0.1
    loss = 1

    Z = [None] * 2
    O = [None] * 2
    while loss >= 0.01:
        #print("yess")
        for i in range(2):
            if i == 0:
                Z[i] = np.dot(X_train, W[i]) + B[i]
                O[i] = sigmoid(Z[i])
            else:
                Z[i] = np.dot(O[i - 1], W[i]) + B[i]
                O[i] = sigmoid(Z[i])

        error = y_train.to_numpy().reshape(-1, 1) - O[len(O) - 1]
        output_error = error * sigmoid_derivative(O[len(O) - 1])

        loss = np.mean((y_train.to_numpy().reshape(-1, 1) - O[len(O) - 1]) ** 2)

        error_hidden = output_error.dot(W[len(W) - 1].T)
        hidden_error = error_hidden * sigmoid_derivative(O[0])

        for i in range(1, -1, -1):
            if i == 1:
                W[i] += O[i - 1].T.dot(output_error) * learning_rate
                B[i] += np.sum(output_error, axis=0, keepdims=True) * learning_rate
            else:
                W[i] += X_train.T.dot(hidden_error) * learning_rate
                B[i] += np.sum(hidden_error, axis=0, keepdims=True) * learning_rate


    Z_test = [None] * 2
    O_test = [None] * 2
    for i in range(2):
        if i == 0:
            Z_test[i] = np.dot(X_test, W[i]) + B[i]
            O_test[i] = sigmoid(Z_test[i])
        else:
            Z_test[i] = np.dot(O_test[i - 1], W[i]) + B[i]
            O_test[i] = sigmoid(Z_test[i])
    O_test[1] = O_test[1].round()
    correct = sum(1 for i in range(y_test.shape[0]) if y_test.iloc[i] == O_test[1][i])
    accuracy = correct / y_test.shape[0]
    result = []
    for i in range(y_test.shape[0]):
        value = O_test[1][i]
        label = 'ckd' if value == 1 else 'notckd'
        result.append({y_test.index[i]: label})

    return accuracy * 100, result
