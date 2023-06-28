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
import datetime

import get_data_update

my_seed = 2023
np.random.seed(my_seed)
random.seed(my_seed)
# tf.random.set_random_seed(my_seed)

tf.random.set_seed(my_seed)

def read_online():
    begin = '2023-05-22 00:00:00'
    end = '2023-06-04 23:59:59'
    pvdataA = get_data_update.get_power_pvA(begin, end)
    pvdataB = get_data_update.get_power_pvB(begin, end)

def add_data(metadata, begintime, endtime):
    time_idx = pd.date_range(start=begintime, end=endtime, freq="H").strftime('%Y-%m-%d %H')
    metadata_df = pd.DataFrame(metadata, index=time_idx)
    start_add = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H') + ':00:00.000'
    end_add = datetime.date.today().strftime('%Y-%m-%d') + ' 23:00:00.000'
    data_add_idx = pd.date_range(start=start_add, end=end_add, freq="H").strftime('%Y-%m-%d %H')
    start_add_past = (datetime.datetime.now() - datetime.timedelta(hours=23)).strftime('%Y-%m-%d %H')
    end_add_past = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d') + ' 23'
    data_add_df = pd.DataFrame((metadata_df.loc[start_add_past:end_add_past, :]).values, index=data_add_idx)
    data_array = np.vstack((metadata_df, data_add_df))
    data = []
    for i in range(data_array.shape[0]):
        data.append(float(data_array[i]))
    return data

def read_from_online(district='B', day_len=50):
    if channel=='A':
        N = 3
        M = 7
    else:
        N = 3
        M = 3
    endtime = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d') + ' 23:00:00.000'
    begintime = (datetime.date.today() + datetime.timedelta(days=-1-M-N-day_len)).strftime('%Y-%m-%d') + ' 00:00:00.000'
    # 获取光伏历史功率
    power = None
    if district == 'A':
        try:
            power = get_data_update.get_power_pvA(begintime, endtime)
        except:
            raise ConnectionError
    if district == 'B':
        try:
            power = get_data_update.get_power_pvB(begintime, endtime)
        except:
            raise ConnectionError
    # 获取历史负荷
    load = None
    if district == 'A1':
        try:
            power = get_data_update.get_load_A_park(begintime, endtime)
        except:
            raise ConnectionError
    if district == 'A2':
        try:
            power = get_data_update.get_load_A_others(begintime, endtime)[2]
        except:
            raise ConnectionError
    if district == 'B1':
        try:
            power = get_data_update.get_load_B_park(begintime, endtime)
        except:
            raise ConnectionError
    if district == 'B2':
        try:
            power = get_data_update.get_load_B_others(begintime, endtime)[2]
        except:
            raise ConnectionError
    # if len(district) ==2:
    #     # 补全历史负荷
    #     #load = add_data(load, begintime, endtime)
    #     # 数据预处理
    #     data = np.array(load).reshape(-1, 24)
    #     df = DataFrame(data, columns=['load' + str(i + 1) for i in range(24)])
    #     dates = np.arange(data.shape[0]) + 1
    #     df_date = DataFrame(dates, columns=['date'])
    #     load_df = concat([df_date, df], axis=1)
    #     # 设置时间戳索引
    #     load_df.set_index("date", inplace=True)
    #
    #     def series_to_supervised(data, n_in=1, dropnan=True):
    #         n_vars = data.shape[1]
    #         df = DataFrame(data)
    #         cols, names = [], []
    #         # i: n_in, n_in-1, ..., 1，为滞后期数
    #         # 分别代表t-n_in, ... ,t-1期
    #         for i in range(n_in, 0, -1):
    #             cols.append(df.shift(i))
    #             names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    #         cols.append(df)
    #         names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    #         agg = concat(cols, axis=1)
    #         agg.columns = names
    #         if dropnan:
    #             agg.dropna(inplace=True)
    #         return agg
    #     def data_process_load(load, n_in):
    #         values = load.values
    #         # 保证所有数据都是float32类型
    #         values = values.astype('float32')
    #         # 变量归一化
    #         scaler = MinMaxScaler(feature_range=(0, 1))
    #         scaled = scaler.fit_transform(values)
    #         #scaled = scaled.fillna(0)
    #         # 将时间序列问题转化为监督学习问题
    #         reframed = series_to_supervised(scaled, n_in=n_in)
    #         # 取出保留的变量
    #         n_vars = values.shape[1]
    #         contain_vars = []
    #         for i in range(1, n_in):
    #             contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars + 1)]
    #         data = reframed[[('var%d(t)' % k) for k in range(1, n_vars + 1)] + contain_vars]
    #         test = data.values
    #         # 将输入X改造为LSTM的输入格式，即[samples,timesteps,features]
    #         input = test.reshape((test.shape[0], n_in, n_vars))
    #         return scaler, input
    #     data_prepare = data_process_load(load_df, n_in=3)
    #     return data_prepare
    # else:
    # 将当天缺少的数据补齐
    #power = add_data(power, begintime, endtime)
    # 获取气象数据（到预测当天）
    #endtime = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d') + ' 23:00:00.000'
    if channel == 'A' or channel == 'B':
        power = list(map(lambda x: 0 if x < 50 else x, power))
    else:
        power = list(map(lambda x: 0 if x<0 else x, power))
    #power = list(map(lambda x: 500 if x > 500 else x, power))
    try:
        weather_temp = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000001')
        weather_humi = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000002')
        weather_type = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000006')
    except:
        raise ConnectionError

    # 数据预处理
    # power += [None for i in range(24)]  # 补齐power
    time_idx = pd.date_range(start=begintime, end=endtime, freq="H")
    week_day = time_idx.day_of_week
    from chinese_calendar import is_workday
    work_day = [1 if is_workday(x) else 0 for x in time_idx]
    #data = np.vstack([power, week_day, work_day, weather_temp, weather_humi, weather_type]).reshape(6, -1, 24)
    data = np.vstack([power, work_day, weather_temp, weather_humi, weather_type]).reshape(5, -1, 24)
    def get_vec(input_data, m, n):
        vec = []
        for day in range(m+n-2, input_data.shape[1]):
            for hour in range(input_data.shape[2]):
                for i in range(m, 0, -1):
                    # tmp_vec = [input_data[1, day - i+1, hour], input_data[2, day - i+1, hour], input_data[3, day - i+1, hour],
                    #            input_data[4, day - i+1, hour], input_data[5, day - i+1, hour], hour]
                    tmp_vec = [input_data[1, day - i + 1, hour], input_data[2, day - i + 1, hour],
                               input_data[3, day - i + 1, hour],
                               input_data[4, day - i+1, hour], hour]
                    for j in range(n, 0, -1):
                        tmp_vec.append(input_data[0, day - m - j+2, hour])
                    vec.append(tmp_vec)
        return np.array(vec, dtype=np.float32).reshape((input_data.shape[1] - m-n+2, input_data.shape[2], m, -1))

    final_vec = get_vec(data, M, N) #A 3,9
    return final_vec, time_idx[(M+N-2)*24:]
    # #data = np.vstack([power, weather_temp, weather_humi, weather_type]).reshape(4, -1, 24)
    # vec = []
    # #day = 4
    # for day in range(8, data.shape[1]-1):
    #     for hour in range(data.shape[2]):
    #         vec.append(
    #             [data[1, day - 1, hour], data[2, day - 1, hour], data[3, day - 1, hour], data[4, day - 1, hour], data[5, day - 1, hour], hour, data[0, day - 8, hour], data[0, day - 3, hour],
    #              data[0, day - 2, hour], data[0, day - 1, hour]])
    #         vec.append(
    #             [data[1, day, hour], data[2, day, hour], data[3, day, hour], data[4, day, hour], data[5, day, hour], hour, data[0, day - 7, hour], data[0, day - 2, hour],
    #              data[0, day - 1, hour], data[0, day, hour]])
    #         vec.append(
    #             [data[1, day + 1, hour], data[2, day + 1, hour], data[4, day + 1, hour], data[5, day + 1, hour], data[3, day + 1, hour], hour, data[0, day - 6, hour], data[0, day - 1, hour],
    #              data[0, day, hour], data[0, day + 1, hour]])
    #         # vec.append(
    #         #     [data[1, day - 1, hour], data[2, day - 1, hour], data[3, day - 1, hour], hour, data[0, day - 3, hour],
    #         #      data[0, day - 2, hour], data[0, day - 1, hour]])
    #         # vec.append(
    #         #     [data[1, day, hour], data[2, day, hour], data[3, day, hour],
    #         #      hour, data[0, day - 2, hour],
    #         #      data[0, day - 1, hour], data[0, day, hour]])
    #         # vec.append(
    #         #     [data[1, day + 1, hour], data[2, day + 1, hour],
    #         #      data[3, day + 1, hour], hour, data[0, day - 1, hour],
    #         #      data[0, day, hour], data[0, day + 1, hour]])
    # vec = np.array(vec, dtype=np.float32).reshape((data.shape[1] - 9, data.shape[2], 3, -1))
    # #scaler, values, input = prepare_data(dataset)
    # print(vec.shape)
    # return vec

# def convert():
#     #dataset = read_excel(io='powerdata0321.xlsx', sheet_name='data2')
#     #del dataset[dataset.columns[0]]
#     #data1 = np.array(dataset.values)
#     dataset = read_excel(io='powerdata0331.xlsx', sheet_name='data2')
#     data = np.delete(np.array(dataset.values), 3, 0).reshape(4, -1, 24)
#     #data = np.hstack([data1, data2]).reshape(4, -1, 24)
#     print(data.shape)# 4天为一个周期，
#     vec = []
#     for day in range(4, data.shape[1]):
#         for hour in range(data.shape[2]):
#             vec.append([data[1, day - 2, hour], data[2, day - 2, hour], data[3, day - 2, hour], hour, data[0, day - 4, hour], data[0, day - 3, hour], data[0, day - 2, hour]])
#             vec.append([data[1, day - 1, hour], data[2, day - 1, hour], data[3, day - 1, hour], hour, data[0, day - 3, hour], data[0, day - 2, hour], data[0, day - 1, hour]])
#             vec.append([data[1, day, hour], data[2, day, hour], data[3, day, hour], hour, data[0, day - 2, hour], data[0, day - 1, hour], data[0, day, hour]])
#     vec = np.array(vec, dtype=np.float32).reshape((data.shape[1] - 4, data.shape[2], 3, -1))
#     print(vec.shape)
#     return vec


def prepare_data(dataset, test_num=10):
    time_index = dataset[1]
    dataset = dataset[0]
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
    return scaler, values, train_X, train_y, test_X, test_y, dataset, time_index[:n_train*24], time_index[n_train*24:]

def prepare_data_load(dataset, test_num=10):
    scaler = dataset[0]
    n_day, n_history, n_hour = dataset[1].shape
    values = dataset[1]
    #分隔数据集，分为训练集和测试集
    n_train = n_day - test_num
    train = values[:n_train, :, :]
    test = values[n_train:, :, :]
    #分隔输入X和输出y, [samples,timesteps,features]
    train_X, train_y = train[:, :-1, :], train[:, -1, :]
    test_X, test_y = test[:, :-1, :], test[:, -1, :]
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
    return scaler, values, train_X, train_y, test_X, test_y, dataset[1]


def fit_lstm(data_prepare, n_batch=24, n_epoch=100, n_neurons=50, loss='mae', optimizer='adam'):
    train_X = data_prepare[2]
    train_y = data_prepare[3]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    #设计神经网络
    model = Sequential()
    model.add(LSTM(48, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=True, input_shape=(train_X.shape[1], 96)))
    # model.add(Dropout(0.4))
    # model.add(LSTM(8, return_sequences=True, input_shape=(train_X.shape[1], 96)))
    # model.add(Dropout(0.4))
    model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[1]))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(loss=loss, optimizer=optimizer)
    #拟合神经网络
    history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    return model


def lstm_predict(model, data_prepare):
    scaler = data_prepare[0]
    train_X = data_prepare[2]
    train_y = data_prepare[3]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    #做出预测
    yhat = model.predict(test_X).clip(0, 1)
    yhat_train = model.predict(train_X).clip(0, 1)
    print(root_mean_squared_error(test_y, yhat))
    # 还原为原来的数据维度
    scaler_new = MinMaxScaler()
    scaler_new.min_, scaler_new.scale_ = scaler.min_[-1], scaler.scale_[-1]
    inv_yhat = scaler_new.inverse_transform(yhat.flatten().reshape(-1, 1))
    inv_y = scaler_new.inverse_transform(test_y.flatten().reshape(-1, 1))
    inv_yhat_train = scaler_new.inverse_transform(yhat_train.flatten().reshape(-1, 1))
    inv_y_train = scaler_new.inverse_transform(train_y.flatten().reshape(-1, 1))
    return inv_yhat, inv_y, inv_yhat_train, inv_y_train


# 计算真实值和预测值的RMSE
def root_mean_squared_error(y_true, y_pred):
    rmse_batch = np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))
    return np.mean(rmse_batch)


if __name__ == '__main__':

    # 数据
    # dataset = convert()
    for channel in ['A', 'A1', 'A2', 'B', 'B1', 'B2']:
        #channel='B1'
        dataset = read_from_online(district=channel)


        n_epoch = 160
        n_batch = 24

        data_prepare = prepare_data(dataset)
        time_index = data_prepare[-2:]
        data_prepare = data_prepare[:-2]

        # if len(channel) == 1:
        #     data_prepare = prepare_data(dataset)
        # else:
        #     data_prepare = prepare_data_load(dataset)
        # scaler, data, train_X, train_y, test_X, test_y, dataset = data_prepare

        # 模型
        model = fit_lstm(data_prepare, n_batch, n_epoch)
        inv_yhat, inv_y, train_yhat, train_y = lstm_predict(model, data_prepare)
        rmse = root_mean_squared_error(inv_y, inv_yhat)
        test_df = pd.DataFrame(list(inv_yhat[:,0]))
        test_df.columns = ['y_hat']
        test_df['y'] = inv_y
        test_df['time_index'] = time_index[1]
        test_df['day'] = test_df['time_index'].apply(lambda x: x.day)
        test_df['month'] = test_df['time_index'].apply(lambda x: x.month)
        test_df['abs_error_rate'] = test_df.apply(lambda x: np.abs(x['y_hat'] - x['y'])/(0.001+x['y_hat']), axis=1)
        test_df['abs_error'] = test_df.apply(lambda x: np.abs(x['y_hat'] - x['y']), axis=1)
        print('test:', test_df.groupby(['month','day']).mean()[['abs_error', 'y_hat', 'y', 'abs_error_rate']])
        test_df = pd.DataFrame(list(train_yhat[:, 0]))
        test_df.columns = ['y_hat']
        test_df['y'] = train_y
        test_df['time_index'] = time_index[0]
        test_df['day'] = test_df['time_index'].apply(lambda x: x.day)
        test_df['month'] = test_df['time_index'].apply(lambda x: x.month)
        test_df['abs_error_rate'] = test_df.apply(lambda x: np.abs(x['y_hat'] - x['y']) / (0.001 + x['y_hat']), axis=1)
        test_df['abs_error'] = test_df.apply(lambda x: np.abs(x['y_hat'] - x['y']), axis=1)
        print('train:', test_df.groupby(['month', 'day']).mean()[['abs_error', 'y_hat', 'y', 'abs_error_rate']])
        plt.plot(inv_yhat, label='yhat')
        plt.plot(inv_y, label='y')
        plt.legend()

        plt.savefig(f'test_40_{channel}.png')
        plt.close()
        plt.plot(train_yhat, label='train_yhat')
        plt.plot(train_y, label='train_y')
        plt.legend()
        plt.savefig(f'train_40_{channel}.png')
        plt.close()

        print("epoch%i_batch%i (rmse: %f)" % (n_epoch, n_batch, rmse))
        model.save_weights(f'best_weights_{channel}.h5')

