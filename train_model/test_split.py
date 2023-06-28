import os
import numpy as np
import random 
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_excel
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
import tensorflow as tf

my_seed = 2023
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_random_seed(my_seed)


def convert():
    dataset = read_excel(io='powerdata0321.xlsx', sheet_name='data2')
    del dataset[dataset.columns[0]]
    data1 = np.array(dataset.values)
    dataset = read_excel(io='powerdata0331.xlsx', sheet_name='data2')
    data2 = np.delete(np.array(dataset.values), 3, 0)
    data = np.hstack([data1, data2]).reshape(4, -1, 24)
    print(data.shape)
    vec = []
    for day in range(4, data.shape[1]):
        for hour in range(data.shape[2]):
            vec.append([data[1, day - 2, hour], data[2, day - 2, hour], data[3, day - 2, hour], hour, data[0, day - 4, hour], data[0, day - 3, hour], data[0, day - 2, hour]])
            vec.append([data[1, day - 1, hour], data[2, day - 1, hour], data[3, day - 1, hour], hour, data[0, day - 3, hour], data[0, day - 2, hour], data[0, day - 1, hour]])
            vec.append([data[1, day, hour], data[2, day, hour], data[3, day, hour], hour, data[0, day - 2, hour], data[0, day - 1, hour], data[0, day, hour]])
    vec = np.array(vec, dtype=np.float32).reshape((data.shape[1] - 4, data.shape[2], 3, -1))
    print(vec.shape)
    return vec


def prepare_data(dataset, test_num=10):
    n_day, n_hour, n_history, n_feature = dataset.shape
    #变量归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset.reshape((n_day * n_hour * n_history, n_feature)))
    values = scaled.reshape((n_day, n_hour, n_history, n_feature))
    #分隔数据集，分为训练集和测试集
    n_train = n_day - test_num
    train = values[:n_train, :, :, :].reshape((n_train * n_hour, n_history, n_feature))
    test = values[n_train:, :, :, :].reshape(((n_day - n_train) * n_hour, n_history, n_feature))
    #分隔输入X和输出y, [samples,timesteps,features]
    train_X, train_y = train[:, :, :-1], train[:, -1, -1, np.newaxis]
    test_X, test_y = test[:, :, :-1], test[:, -1, -1, np.newaxis]
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
    return scaler, values, train_X, train_y, test_X, test_y, dataset

 
def lstm():
    #设计神经网络
    model = Sequential()
    model.add(LSTM(48, return_sequences=True, input_shape=(3, 6)))
    model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
    model.add(Dense(16))
    model.add(Dense(1))
    return model


def lstm_predict(model, scaler, test_X, test_y):
    #做出预测
    yhat = model.predict(test_X).clip(0, 1)
    for i in range(test_y.shape[0]):
        hour = np.round(test_X[i, -1, 3] * 23)
        if hour <= 6 or hour >= 18:
            yhat[i] = 0

    # 还原为原来的数据维度
    scaler_new = MinMaxScaler()
    scaler_new.min_, scaler_new.scale_ = scaler.min_[-1], scaler.scale_[-1]
    inv_yhat = scaler_new.inverse_transform(yhat)
    inv_y = scaler_new.inverse_transform(test_y)
    return inv_yhat, inv_y


# 计算真实值和预测值的RMSE
def root_mean_squared_error(y_true, y_pred):
    rmse_batch = np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))
    return np.mean(rmse_batch)


if __name__ == '__main__':

    # 模型
    model = lstm()
    model.load_weights('best_weights.h5')

    # 数据
    dataset = convert()

    data_prepare = prepare_data(dataset)
    scaler, data, train_X, train_y, test_X, test_y, dataset = data_prepare
    
    inv_yhat, inv_y = lstm_predict(model, scaler, test_X, test_y)
    rmse_11 = root_mean_squared_error(inv_y.reshape((-1, 24, 1))[:, 7:18, :].flatten()[:, np.newaxis], inv_yhat.reshape((-1, 24, 1))[:, 7:18, :].flatten()[:, np.newaxis])
    rmse = root_mean_squared_error(inv_y, inv_yhat)
    print(rmse_11, rmse)

    plt.figure()
    plt.plot(np.arange(len(inv_y)), inv_y, color='black', label='true', linewidth=1)
    plt.plot(np.arange(len(inv_y)), inv_yhat, color='red', label='predict', linewidth=1)
    plt.ylabel('pv')
    plt.xlabel('day(24H)')
    plt.title('Test Set (RMSE: %.2f)' % rmse)
    plt.legend()
    plt.savefig('result_test.png')
    plt.close()

    inv_yhat, inv_y = lstm_predict(model, scaler, train_X, train_y)
    rmse_11 = root_mean_squared_error(inv_y.reshape((-1, 24, 1))[:, 7:18, :].flatten()[:, np.newaxis], inv_yhat.reshape((-1, 24, 1))[:, 7:18, :].flatten()[:, np.newaxis])
    rmse = root_mean_squared_error(inv_y, inv_yhat)
    print(rmse_11, rmse)

    plt.figure()
    plt.plot(np.arange(len(inv_y)), inv_y, color='black', label='true', linewidth=1)
    plt.plot(np.arange(len(inv_y)), inv_yhat, color='red', label='predict', linewidth=1)
    plt.ylabel('pv')
    plt.xlabel('day(24H)')
    plt.title('Train Set (RMSE: %.2f)' % rmse)
    plt.legend()
    plt.savefig('result_train.png')
    plt.close()


