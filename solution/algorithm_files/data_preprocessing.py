import random

import pandas as pd
import numpy as np


def cat_to_num(data):
    for col in data:
        if (col not in ['appet', 'classification']) and (data[col].dtype == 'object'):
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes

    mapping = {'good' : 1, 'poor' : 0}
    data['appet'] = data['appet'].replace(mapping)

    c_mapping = {'ckd': 1, 'ckd\t': 1, 'notckd': 0}
    data['classification'] = data['classification'].replace(c_mapping)
    return data

def normalize_data(data):
    for col in data:
        if col != "id" and col != "classification":
            data[col] = ((data[col] - data[col].min()) / (data[col].max() - data[col].min()))
    return data

def check_nulls(data):
    for col in data:
        print(data[col].isnull().sum()/data.shape[0] * 100)

def fill_nulls(data):
    for col in data:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col].fillna(data[col].mean(), inplace=True)
    return data

def divide_data(data, precentage):
    data = data.drop('id', axis = 1)
    alg = random.randint(0,100)
    df_shuffled = data.sample(frac=1, random_state=alg).reset_index(drop=True)

    train_size = None
    if precentage is None:
        train_size = int((70 / 100) * len(df_shuffled))
    else:
        train_size = int((precentage/100) * len(df_shuffled))

    print(train_size)

    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]

    X_train = train_df.drop('classification', axis=1)
    y_train = train_df['classification']
    X_test = test_df.drop('classification', axis=1)
    y_test = test_df['classification']

    return X_train, y_train, X_test, y_test

def data_transformation(data):
    data = cat_to_num(data)
    data = normalize_data(data)
    return data

def replace_nulls(data):
    data.fillna(0, inplace = True)
    return data

def dataPreprocessing(path, precentage):
    data = pd.read_csv(path)
    #data = data.iloc[:int(data.shape[0] * (precentage / 100))]
    data = fill_nulls(data)
    data = data_transformation(data)
    data['classification'] = data['classification'].astype(int)
    data = replace_nulls(data)
    #check_nulls(data)

    X_train, y_train, X_test, y_test = divide_data(data, precentage)
    #print(y_train.dtype)
    return X_train, y_train, X_test, y_test

#dataPreprocessing('D:\\Collage\\4th_year\\second_semester\\Data Mining\\Assignment(3)\\Kidney_Disease Data for classification.csv',70)