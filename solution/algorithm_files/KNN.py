import data_preprocessing as dp
import numpy as np

# X_train, y_train, X_test, y_test = dp.dataPreprocessing('D:\\Collage\\4th_year\\second_semester\\Data Mining\\Assignment(3)\\Kidney_Disease_data_for_Classification_V2.csv', 70)


def euclidean_distance(record1, record2) -> float:
    vec1 = np.array(record1)
    vec2 = np.array(record2)

    return np.linalg.norm(vec1 - vec2)

def KNN(X_train, y_train, X_test, y_test, k):
    result = []

    for i in range(X_test.shape[0]):
        tempResult = []
        for j in range(X_train.shape[0]):
            tempDis = euclidean_distance(X_test.iloc[i], X_train.iloc[j])
            tempResult.append({'distance': tempDis, 'label': y_train[j]})
        tempResult.sort(key=lambda x: x['distance'])

        ckd = 0
        notckd = 0
        for j in range(k):
            if tempResult[j]['label'] == 0:
                notckd += 1
            else:
                ckd += 1
        if ckd > notckd:
            result.append(1)
        elif notckd > ckd:
            result.append(0)
        else:
            result.append(1)

    correct = sum([1 for i in range(len(y_test)) if y_test.iloc[i] == result[i]])
    accuracy = correct / len(y_test)
    # O_test[1] = np.array(O_test[1].flatten())
    # O_test[1] = np.where(O_test[1] == 1, 'ckd', 'notckd')
    final = []
    for i in range(y_test.shape[0]):
        value = result[i]
        label = 'ckd' if value == 1 else 'notckd'
        final.append({y_test.index[i]: label})
    return accuracy * 100, final
#
# print(KNN(X_train, y_train, X_test, y_test,6))