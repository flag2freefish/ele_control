# -*- coding: utf-8 -*-
# @Time : 2023/4/18 14:10
# @Author : Ziyao Wang
# @FileName: optmization_update.py
# @Describe: optmization_update
# 在线预测部署代码
import configparser
import datetime
import json
import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import pulp
import requests
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
#from tensorflow.python.keras.models import Sequential
from keras.models import Sequential
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

import get_data_update

run_mode = 'remot'

try:
    servicenum = sys.argv[1]
except:
    serviceNum = '010011'
else:
    serviceNum = servicenum
# print('serviceNum', serviceNum)
config = configparser.ConfigParser()
if run_mode == 'local':
    mypath = '/opt/calc/'
    ini_path = '/opt/calc/opt_config.ini'
else:
    ini_path = 'opt_config.ini'
    mypath = ''
config.read(ini_path, encoding='utf-8')

# 本地日志设置
# # 设置日志文件路径
# 获取系统环境变量
delimeter = os.sep
env_path = os.getenv('XQNY_ROOT')
if not env_path:
    if run_mode == 'local':
        env_path = delimeter + 'opt' + delimeter + 'calc' + delimeter + 'logs'
    else:
        env_path = r'D:\ProgramData\mylogs'
# 获取程序名
program_name = os.path.basename(sys.argv[0])
# 生成路径
log_path = env_path + delimeter + program_name
if not os.path.isdir(log_path):
    os.makedirs(log_path)
# 配置日志保存方式、文件名
filename = log_path + delimeter + 'optimization'
formater = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
timehandler = TimedRotatingFileHandler(filename,
                                       when='D',  # 按日分割
                                       interval=1,
                                       backupCount=15,  # 保存15天
                                       encoding='utf-8'
                                       )
timehandler.setFormatter(formater)
timehandler.suffix = "%Y-%m-%d.log"  # 文件日期后缀
logger = logging.getLogger('my_app')
logger.setLevel(logging.INFO)
logger.addHandler(timehandler)

if run_mode == 'remot':
    url0 = config.get('url_remot', 'url_get_cal_tsdb')
    url_get_point = config.get('url_remot', 'url_get_cal_rtdb')
    url_set_ai = config.get('url_remot', 'url_set_user_tsdb_ai')
    url_set_point = config.get('url_remot', 'url_set_user_rtdb')
    url_get_define = config.get('url_remot', 'url_get_define')
else:
    url0 = config.get('url_local', 'url_get_cal_tsdb')
    url_get_point = config.get('url_local', 'url_get_cal_rtdb')
    url_set_ai = config.get('url_local', 'url_set_user_tsdb_ai')
    url_set_point = config.get('url_local', 'url_set_user_rtdb')
    url_get_define = config.get('url_local', 'url_get_define')

outputNum = 28
outputname = list()
for i in range(outputNum):
    outputname.append('output' + str(100 + i)[1:])


def model_define():
    payload = {'serviceNum': serviceNum}
    payload = json.dumps(payload)
    # print('payload', payload)
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url_get_define, headers=headers, data=payload)
    response = response.json()
    # print('response', response)
    if response['data']['data']:
        modelList = response['data']['data'][0]['model']['modelList']
        modelpara_allmodels = list()
        for model_index in range(len(modelList)):
            opttarget = modelList[model_index]['optTarget']['used']  # 经济最优/碳排最优
            optresult_id = list()
            optresult_id.append(modelList[model_index]['optResult'])
            optobject = modelList[model_index]['optObject']['used']  # 全选/1#储能/2#储能
            opttime = modelList[model_index]['optTime']['used']
            optname = modelList[model_index]['optName']['used']
            modelpara_allmodels.append([opttarget, optobject, opttime, optresult_id, optname])
        return [modelpara_allmodels, len(modelList)]
    else:
        return []


my_seed = 2023
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)


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


def lstm_pv():
    # 设计神经网络
    model = Sequential()
    model.add(LSTM(48, return_sequences=True, input_shape=(3, 6)))
    model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
    model.add(Dense(16))
    model.add(Dense(1))
    return model


def convert_pv(power, weather_temp, weather_humi, weather_type):
    power += [None for i in range(24)]  # 补齐power
    data = np.vstack([power, weather_temp, weather_humi, weather_type]).reshape(4, -1, 24)
    vec = []
    day = 3
    for hour in range(data.shape[2]):
        vec.append(
            [data[1, day - 1, hour], data[2, day - 1, hour], data[3, day - 1, hour], hour, data[0, day - 3, hour],
             data[0, day - 2, hour]])
        vec.append(
            [data[1, day, hour], data[2, day, hour], data[3, day, hour], hour, data[0, day - 2, hour],
             data[0, day - 1, hour]])
        vec.append(
            [data[1, day + 1, hour], data[2, day + 1, hour], data[3, day + 1, hour], hour, data[0, day - 1, hour],
             data[0, day, hour]])
    vec = np.array(vec, dtype=np.float32).reshape((1, data.shape[2], 3, -1))
    return vec


def prepare_data(dataset):
    n_day, n_hour, n_history, n_feature = dataset.shape
    # 变量归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset.reshape((n_day * n_hour * n_history, n_feature)))
    values = scaled.reshape((n_day, n_hour, n_history, n_feature))
    # 分隔数据集，分为训练集和测试集
    test = values.reshape((n_day * n_hour, n_history, n_feature))
    return scaler, values, test


def lstm_predict_pv(model, scaler, input):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./pv/logs")
    yhat = model.predict(input, callbacks=[tensorboard_callback]).clip(0, 1)
    # 还原为原来的数据维度
    scaler_new = MinMaxScaler()
    scaler_new.min_, scaler_new.scale_ = scaler.min_[-1], scaler.scale_[-1]
    inv_yhat = scaler_new.inverse_transform(yhat)
    for i in range(input.shape[0]):
        hour = np.round(input[i, -1, 3] * 23)
        if hour <= 6 or hour >= 18:
            inv_yhat[i] = 0
    return inv_yhat


def convert_load(load):
    data = np.array(load).reshape(-1, 24)
    df = DataFrame(data, columns=['load' + str(i + 1) for i in range(24)])
    dates = np.arange(data.shape[0]) + 1
    df_date = DataFrame(dates, columns=['date'])
    df = concat([df_date, df], axis=1)
    # 设置时间戳索引
    df.set_index("date", inplace=True)
    return df


def series_to_supervised(data, n_in=1, dropnan=True):
    n_vars = data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # i: n_in, n_in-1, ..., 1，为滞后期数
    # 分别代表t-n_in, ... ,t-1期
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    cols.append(df)
    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def data_process_load(load, n_in):
    values = load.values
    # 保证所有数据都是float32类型
    values = values.astype('float32')
    # 变量归一化
    scaler = [np.min(values, axis=0), np.max(values, axis=0)]
    scaled = load.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    scaled = scaled.fillna(0)
    # 将时间序列问题转化为监督学习问题
    reframed = series_to_supervised(scaled, n_in=n_in)
    # 取出保留的变量
    n_vars = values.shape[1]
    contain_vars = []
    for i in range(1, n_in):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars + 1)]
    data = reframed[[('var%d(t)' % k) for k in range(1, n_vars + 1)] + contain_vars]
    test = data.values
    # 将输入X改造为LSTM的输入格式，即[samples,timesteps,features]
    input = test.reshape((test.shape[0], n_in, n_vars))
    return scaler, input


def lstm_load(n_batch=1):
    # 设计神经网络
    model = Sequential()
    model.add(LSTM(48, return_sequences=True, input_shape=(3, 24)))
    model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=False))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.2))
    model.add(Dense(24))
    return model


def lstm_predict_load(model, data_prepare):
    scaler = data_prepare[0]
    input = data_prepare[1]
    # 做出预测
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./load/logs")
    yhat = model.predict(input, callbacks=[tensorboard_callback])
    # 将测试集上的预测值还原为原来的数据维度
    inv_yhat = yhat * (scaler[1] - scaler[0]) + scaler[0]
    return inv_yhat


# 光伏预测
def pv_power_forecast(district):
    # 定义起止时间
    endtime = datetime.datetime.now().strftime('%Y-%m-%d %H') + ':00:00.000'
    begintime = (datetime.date.today() + datetime.timedelta(days=-3)).strftime('%Y-%m-%d') + ' 00:00:00.000'
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
    # 将当天缺少的数据补齐
    power = add_data(power, begintime, endtime)
    # 获取气象数据（到预测当天）
    endtime = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d') + ' 23:00:00.000'
    try:
        weather_temp = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000001')
        weather_humi = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000002')
        weather_type = get_data_update.get_userdb(begintime, endtime, '1h', 'eqm000000', 'QXZ000006')
    except:
        raise ConnectionError

    # 预测光伏功率
    model = lstm_pv()
    model.load_weights('pv_power_forecast.h5')
    # 数据预处理
    dataset = convert_pv(power, weather_temp, weather_humi, weather_type)
    scaler, values, input = prepare_data(dataset)

    # 做出预测
    power_hat = lstm_predict_pv(model, scaler, input)
    yhat = []
    for i in range(power_hat.shape[0]):
        yhat.append(float(power_hat[i]))
    return yhat


# 负荷预测
def load_forecast(load_type):
    # 定义起止时间
    endtime = datetime.datetime.now().strftime('%Y-%m-%d %H') + ':00:00.000'
    begintime = (datetime.date.today() + datetime.timedelta(days=-3)).strftime('%Y-%m-%d') + ' 00:00:00.000'
    # 获取历史负荷
    load = None
    if load_type == 'A1':
        try:
            load = get_data_update.get_load_A_park(begintime, endtime)
        except:
            raise ConnectionError
    if load_type == 'A2':
        try:
            load = get_data_update.get_load_A_others(begintime, endtime)[2]
        except:
            raise ConnectionError
    if load_type == 'B1':
        try:
            load = get_data_update.get_load_B_park(begintime, endtime)
        except:
            raise ConnectionError
    if load_type == 'B2':
        try:
            load = get_data_update.get_load_B_others(begintime, endtime)[2]
        except:
            raise ConnectionError
    # 补全历史负荷
    load = add_data(load, begintime, endtime)
    # 数据预处理
    load_df = convert_load(load)
    data_prepare = data_process_load(load_df, n_in=3)
    model = lstm_load(n_batch=1)
    model.load_weights('load_forecast.h5')
    # 负荷预测
    inv_yhat = lstm_predict_load(model, data_prepare).reshape(24, 1)
    yhat = []
    for i in range(inv_yhat.shape[0]):
        yhat.append(float(inv_yhat[i]))
    return yhat


def set_zero(size):
    data = np.zeros((size, 1))
    yhat = []
    for i in range(data.shape[0]):
        yhat.append(float(data[i]))
    return yhat


# 自定义列表加法
def myplusfunc(listdata):
    number = len(listdata)
    # 校核各数据的长度是否一致
    datalength = len(listdata[0])
    for i in range(len(listdata)):
        if datalength != len(listdata[i]):
            raise SyntaxError
    resultlist = list()
    for j in range(datalength):
        result = 0
        for i in range(len(listdata)):
            result += listdata[i][j]
        resultlist.append(result)
    return resultlist


def myinterpolation(inputdata, sign):  # sign=96 or sign = 288
    hourdata = inputdata
    hourdata.append(inputdata[0])
    if sign == '288':
        data288 = list()
        # print(hourdata, 'len:hourdata288')
        for i in range(len(hourdata) - 1):
            for j in range(12):
                data288.append(hourdata[i] + j * (hourdata[i + 1] - hourdata[i]) / 12)
        inputdata.pop()
        return data288
    elif sign == '96':
        data96 = list()
        # print(hourdata, 'len:hourdata96')
        for i in range(len(hourdata) - 1):
            for j in range(4):
                data96.append(hourdata[i] + j * (hourdata[i + 1] - hourdata[i]) / 4)
        inputdata.pop()
        return data96


soc1_key = config.get('keys_es', 'soc1_key')
soc2_key = config.get('keys_es', 'soc2_key')


def getsoc():
    para = '[' + '"' + soc1_key + '"' + ',' + '"' + soc2_key + '"' + ']'
    querystring = "jsonStr=" + para
    url = url_get_point + "?" + querystring
    # print(url)
    response = requests.get(url)
    response = response.json()
    if response['data']:
        soc1 = response['data'][0]['v']
        soc2 = response['data'][1]['v']
        return [soc1, soc2]
    else:
        raise ConnectionError


Pres_1 = 250
Eres_1 = 500
Pres_2 = 250
Eres_2 = 500
Price_perchase = np.array([0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784,
                           0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784,
                           0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784,
                           0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784,
                           1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439,
                           1.1439, 1.1439, 1.1439, 1.1439, 0.6652, 0.6652, 0.6652, 0.6652,
                           0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652,
                           0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652,
                           0.6652, 0.6652, 0.6652, 0.6652, 1.1439, 1.1439, 1.1439, 1.1439,
                           1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439,
                           1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439, 1.1439,
                           0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652, 0.6652])
Price_sold = 0.3


# 1、经济最优模型
def chenbenyouhua(Pload, Pne, soc0, pres, eres, T):
    c1 = Price_perchase # 每15分钟一个电价
    c2 = Price_sold # 这个是储能的成本吗？多余电量卖出的价格？

    Z1 = np.zeros([96])
    Z2 = np.zeros([96])
    for i in range(0, 95):
        if Pload[i] >= Pne[i]:
            Z1[i] = (Pload[i] - Pne[i]) * c1[i]
        else:
            Z2[i] = (-Pload[i] + Pne[i]) * c2
    Z = (np.sum(Z1) - np.sum(Z2)) / 4 #预测的负荷与出力的电价和的差值，正值表示还需这么多钱，负值表示可以收入这么多钱
    SOCmin = 0.1  # 储能最小荷电状态
    SOCmax = 0.9  # 储能最大荷电状态
    Nch = 0.9  # 储能充电效率
    Ndis = 0.9  # 储能放电效率
    qn = 0.604  # 碳排放因子
    Pch = [pulp.LpVariable(f'Pch{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pdis = [pulp.LpVariable(f'Pdis{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pbuy = [pulp.LpVariable(f'Pbuy{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Psell = [pulp.LpVariable(f'Psell{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    f = pulp.LpProblem("chengben", sense=pulp.const.LpMinimize)#优化目标是最小值，对的，买电成本-售电收益，最终成本，没有电池消耗成本
    sum = (np.array(Pbuy) * np.array(c1) - np.array(Psell) * c2) / 4
    f += pulp.lpSuml(sum)
    for i in range(0, int(T)):
        f += Pne[i] + Pbuy[i] + Pdis[i] == Pload[i] + Psell[i] + Pch[i]# 将储能放电加入限制
        f += Pch[i] <= pres
        f += Pdis[i] <= pres
    Eres1 = 1 / eres
    Ndis1 = 1 / Ndis
    for i in range(0, 96):
        Eres1 = 1 / eres
        Ndis1 = 1 / Ndis
        f += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4 >= SOCmin
        f += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4 <= SOCmax
    #TODO 确认 充电和放电的总和相等 = np.sum(Pch[0:int(T)])==np.sum(Pdis[0:int(T)])，这个为啥要相等，不一定相等吧？ 这个soc是储能的soc，目前是假数据
    f += soc0 + (np.sum(Pch[0:int(T)]) * Nch * Eres1 - np.sum(Pdis[0:int(T)]) * Ndis1 * Eres1) / 4 == soc0
    f.solve()
    # for i in f.variables():
    #     print(i.name,"=",i.varValue)
    for i in range(0, int(T)):
        Pch[i] = Pch[i].varValue
        Pdis[i] = Pdis[i].varValue
        Pbuy[i] = Pbuy[i].varValue
        Psell[i] = Psell[i].varValue
    f = pulp.value(f.objective) #f为加入储能设备之后，需要买电
    SOC = np.empty([96])
    for i in range(0, int(T)):
        SOC[i] = soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4
    c = qn * np.sum(Pbuy[0:96]) / 4 #买点的碳排，成本优先下，还是会计算这个。
    Pro = Z - f # 无储能设备和加入储能设备后的变化
    Pes = np.array(Pdis) - np.array(Pch)
    Pes_list = Pes.tolist()
    SOC_list = SOC.tolist()
    return Pes_list, SOC_list, [f, c, Pro]


# 2、碳排最优模型
# 碳排优化函数-第一层：得到c值
def tanpaiyouhua_1(Pload, Pne, soc0, pres, eres, T):
    SOCmin = 0.1  # 储能最小荷电状态
    SOCmax = 0.9  # 储能最大荷电状态
    Nch = 0.9  # 储能充电效率
    Ndis = 0.9  # 储能放电效率
    qn = 0.604  # 碳排放因子
    Pch = [pulp.LpVariable(f'Pch{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pdis = [pulp.LpVariable(f'Pdis{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pbuy = [pulp.LpVariable(f'Pbuy{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Psell = [pulp.LpVariable(f'Psell{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    c = pulp.LpProblem("tanpai", sense=pulp.const.LpMinimize)
    c += pulp.lpSum(Pbuy) * qn / 4 # 碳排最小
    for i in range(0, int(T)):
        c += Pne[i] + Pbuy[i] + Pdis[i] == Pload[i] + Psell[i] + Pch[i]
        c += Pch[i] <= pres
        c += Pdis[i] <= pres
    Eres1 = 1 / eres
    Ndis1 = 1 / Ndis
    for i in range(0, 96):
        Eres1 = 1 / eres
        Ndis1 = 1 / Ndis
        c += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) >= SOCmin
        c += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) <= SOCmax
    c += soc0 + (np.sum(Pch[0:int(T)]) * Nch * Eres1 - np.sum(Pdis[0:int(T)]) * Ndis1 * Eres1) / 4 == soc0
    c.solve()
    for i in range(0, int(T)):
        Pch[i] = Pch[i].varValue
        Pdis[i] = Pdis[i].varValue
        Pbuy[i] = Pbuy[i].varValue
        Psell[i] = Psell[i].varValue
    c = pulp.value(c.objective)
    return Pbuy, c


# 碳排优化函数-第二层：输入c值优化
def tanpaiyouhua_2(Pload, Pne, soc0, pres, eres, T, c):
    c1 = Price_perchase
    c2 = Price_sold

    Z1 = np.zeros([96])
    Z2 = np.zeros([96])
    for i in range(0, 95):
        if Pload[i] >= Pne[i]:
            Z1[i] = (Pload[i] - Pne[i]) * c1[i]
        else:
            Z2[i] = (-Pload[i] + Pne[i]) * c2
    Z = (np.sum(Z1) - np.sum(Z2)) / 4
    SOCmin = 0.1  # 储能最小荷电状态
    SOCmax = 0.9  # 储能最大荷电状态
    Nch = 0.9  # 储能充电效率
    Ndis = 0.9  # 储能放电效率
    qn = 0.604  # 碳排放因子
    Pch = [pulp.LpVariable(f'Pch{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pdis = [pulp.LpVariable(f'Pdis{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Pbuy = [pulp.LpVariable(f'Pbuy{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    Psell = [pulp.LpVariable(f'Psell{i}', lowBound=0, upBound=None) for i in range(0, 96)]
    f = pulp.LpProblem("chengben", sense=pulp.const.LpMinimize)
    sum = (np.array(Pbuy) * np.array(c1) - np.array(Psell) * c2) / 4
    f += pulp.lpSum(sum)
    for i in range(0, int(T)):
        f += Pne[i] + Pbuy[i] + Pdis[i] == Pload[i] + Psell[i] + Pch[i]
        f += Pch[i] <= pres
        f += Pdis[i] <= pres
    Eres1 = 1 / eres
    Ndis1 = 1 / Ndis
    for i in range(0, 96):
        Eres1 = 1 / eres
        Ndis1 = 1 / Ndis
        f += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4 >= SOCmin
        f += soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4 <= SOCmax
    f += soc0 + (np.sum(Pch[0:int(T)]) * Nch * Eres1 - np.sum(Pdis[0:int(T)]) * Ndis1 * Eres1) / 4 == soc0
    qn1 = 1 / qn
    f += np.sum(Pbuy[0:int(T) - 1]) == c * 4 * qn1 #加入优化限制，在碳排等于最低碳排指标的前提下，寻找成本最低的目标，两层优化
    f.solve()
    for i in range(0, int(T)):
        Pch[i] = Pch[i].varValue
        Pdis[i] = Pdis[i].varValue
        Pbuy[i] = Pbuy[i].varValue
        Psell[i] = Psell[i].varValue
    f = pulp.value(f.objective)
    SOC = np.empty([96])
    for i in range(0, int(T)):
        SOC[i] = soc0 + (np.sum(Pch[0:i + 1]) * Nch * Eres1 - np.sum(Pdis[0:i + 1]) * Ndis1 * Eres1) / 4
    c = qn * np.sum(Pbuy[0:96]) / 4
    Pro = Z - f
    Pes = np.array(Pdis) - np.array(Pch)

    Pes_list = Pes.tolist()
    SOC_list = SOC.tolist()
    return Pes_list, SOC_list, [f, c, Pro]


def optimize1(optinput, opttarget, amount):  # optTarget = 经济最优/碳排最优 # optinput=[load,power,soc0]
    if opttarget == '经济最优':
        Pes, soclist, profit = chenbenyouhua(optinput[0], optinput[1], optinput[2], Pres_1, Eres_1, 96)
    else:
        pbuy, carbon_profit = tanpaiyouhua_1(optinput[0], optinput[1], optinput[2], Pres_1, Eres_1, 96)
        Pes, soclist, profit = tanpaiyouhua_2(optinput[0], optinput[1], optinput[2], Pres_1, Eres_1, 96, carbon_profit)
    output_stra = Pes # 原始结果是15分钟一个点
    optout_profit = profit
    output_stra24 = list()
    for i in range(24):# 按24小时取点，应该是取得整点时刻，或以开头时刻为头，间隔1小时的24个点
        output_stra24.append(output_stra[4 * i])
    output_stra288 = myinterpolation(output_stra24, '288') # 插值成5分钟一个点，目标是1分钟一个点
    if amount == 96:
        return output_stra, optout_profit
    elif amount == 24:
        return output_stra24, optout_profit
    else:
        return output_stra288, optout_profit


def optimize2(optinput, opttarget, amount):  # optTarget = 经济最优/碳排最优
    if opttarget == '经济最优':
        Pes, soclist, profit = chenbenyouhua(optinput[0], optinput[1], optinput[2], Pres_2, Eres_2, 96)
    else:
        pbuy, carbon_profit = tanpaiyouhua_1(optinput[0], optinput[1], optinput[2], Pres_2, Eres_2, 96)
        Pes, soclist, profit = tanpaiyouhua_2(optinput[0], optinput[1], optinput[2], Pres_2, Eres_2, 96, carbon_profit)
    output_stra = Pes
    optout_profit = profit
    output_stra24 = list()
    for i in range(24):
        output_stra24.append(output_stra[4 * i])
    output_stra288 = myinterpolation(output_stra24, '288')
    if amount == 96:
        return output_stra, optout_profit
    elif amount == 24:
        return output_stra24, optout_profit
    else:
        return output_stra288, optout_profit


hourpool = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
minutepool = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
minutepool2 = ['00', '15', '30', '45']
timestamp_list = list()
for hour in range(24):
    for minute in range(12):
        timestamp_list.append(hourpool[hour] + ':' + minutepool[minute] + ':00.000')

timestamp_list96 = list()
for hour in range(24):
    for minute in range(4):
        timestamp_list96.append(hourpool[hour] + ':' + minutepool2[minute] + ':00.000')

timestamp_list24 = list()
for hour in range(24):
    timestamp_list24.append(hourpool[hour] + ':00:00.000')


def post_optresult_A(load, power, es, profit, userid, amount):
    data_list = [load[0], load[1], load[2], load[3], load[4], power[0], power[1], es]
    userid_list = [userid['output01'], userid['output04'], userid['output05'], userid['output03'], userid['output02'],
                   userid['output11'], userid['output19'], userid['output21']]
    profit_userid = [userid['output22'], userid['output23'], userid['output24']]
    eqmnum = 'eqm000000'
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if amount == 96:
        stamplist = timestamp_list96
    elif amount == 24:
        stamplist = timestamp_list24
    else:
        stamplist = timestamp_list

    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    payload2 = {}
    for k in range(len(profit)):
        key = eqmnum + ':ai:' + profit_userid[k]
        payload2[key] = {'v': str(profit[k]),
                         't': tomorrow + ' ' + '23:59:59.000'}
    payload2 = json.dumps(payload2)
    requests.request("POST", url_set_ai, data=payload2, headers=headers)
    requests.request("POST", url_set_point, data=payload2, headers=headers)

    for j in range(len(stamplist)):
        payload_rt = {}
        payload_ts = list()
        for i in range(len(data_list)):
            # print('timenum=', j, 'datanum=', i)
            key = eqmnum + ':ai:' + userid_list[i]
            payload = {'eqmid': eqmnum + ':ai:' + userid_list[i],
                       'v': str(data_list[i][j]),
                       't': tomorrow + ' ' + stamplist[j]}
            payload_ts.append(payload)
            payload_rt[key] = {'v': str(data_list[i][j]),
                               't': tomorrow + ' ' + stamplist[j]}
        payload_rt = json.dumps(payload_rt)
        payload_ts = json.dumps(payload_ts)
        requests.request("POST", url_set_ai, data=payload_ts, headers=headers)
        requests.request("POST", url_set_point, data=payload_rt, headers=headers)


def post_optresult_B(load, power, es, profit, userid, amount):
    data_list = [load[0], load[1], load[2], load[3], load[4], power, es]
    userid_list = [userid['output06'], userid['output09'], userid['output10'], userid['output08'], userid['output07'],
                   userid['output12'], userid['output25']]
    profit_userid = [userid['output26'], userid['output27'], userid['output28']]
    eqmnum = 'eqm000000'
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if amount == 96:
        stamplist = timestamp_list96
    elif amount == 24:
        stamplist = timestamp_list24
    else:
        stamplist = timestamp_list

    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    payload2 = {}
    for k in range(len(profit)):
        key = eqmnum + ':ai:' + profit_userid[k]
        payload2[key] = {'v': str(profit[k]),
                         't': tomorrow + ' ' + '23:59:59.000'}
    payload2 = json.dumps(payload2)
    requests.request("POST", url_set_ai, data=payload2, headers=headers)
    requests.request("POST", url_set_point, data=payload2, headers=headers)

    for j in range(len(stamplist)):
        payload_rt = {}
        payload_ts = list()
        for i in range(len(data_list)):
            # print('timenum=', j, 'datanum=', i)
            key = eqmnum + ':ai:' + userid_list[i]
            payload = {'eqmid': eqmnum + ':ai:' + userid_list[i],
                       'v': str(data_list[i][j]),
                       't': tomorrow + ' ' + stamplist[j]}
            payload_ts.append(payload)
            payload_rt[key] = {'v': str(data_list[i][j]),
                               't': tomorrow + ' ' + stamplist[j]}
        payload_rt = json.dumps(payload_rt)
        payload_ts = json.dumps(payload_ts)
        requests.request("POST", url_set_ai, data=payload_ts, headers=headers)
        requests.request("POST", url_set_point, data=payload_rt, headers=headers)


if __name__ == '__main__':
    logger.info('程序启动')
    # A区负荷预测结果，分别为园区负荷、充电桩、智慧路灯、直流照明、总负荷；24点
    # begintime0 = time.time()
    loadA_park = load_forecast(load_type='A1')
    loadA_pile = set_zero(24)
    loadA_lude = set_zero(24)
    loadA_zlzm = load_forecast(load_type='A2')
    # print('负荷A预测用时=', time.time() - begintime0)
    # A区新能源出力预测，分别为光伏、风机；24点
    begintime = time.time()
    pvA_power = pv_power_forecast('A')
    windA_power = set_zero(24)
    # print('出力A预测用时=', time.time() - begintime)

    # B区负荷预测结果，分别为园区负荷、充电桩、智慧路灯、直流照明、总负荷；24点
    # begintime = time.time()
    loadB_park = load_forecast(load_type='B1')
    loadB_pile = set_zero(24)
    loadB_lude = set_zero(24)
    loadB_zlzm = load_forecast(load_type='B2')
    # print('负荷B预测用时=', time.time() - begintime)
    # B区新能源出力预测，分别为光伏、风机；24点
    # begintime = time.time()
    pvB_power = pv_power_forecast('B')
    # print('出力B预测用时=', time.time() - begintime)
    logger.info('负荷/出力预测完成')

    output_amount = 24
    loadA_total = myplusfunc([loadA_park, loadA_pile, loadA_lude, loadA_zlzm])
    # A区负荷预测结果，分别为园区负荷、充电桩、智慧路灯、直流照明；288点
    loadA_park288 = myinterpolation(loadA_park, '288')
    loadA_pile288 = myinterpolation(loadA_pile, '288')
    loadA_lude288 = myinterpolation(loadA_lude, '288')
    loadA_zlzm288 = myinterpolation(loadA_zlzm, '288')
    loadA_total288 = myinterpolation(loadA_total, '288')
    loadA_park96 = myinterpolation(loadA_park, '96')
    loadA_pile96 = myinterpolation(loadA_pile, '96')
    loadA_lude96 = myinterpolation(loadA_lude, '96')
    loadA_zlzm96 = myinterpolation(loadA_zlzm, '96')
    loadA_total96 = myinterpolation(loadA_total, '96')
    # A区新能源出力预测，分别为光伏、风机；插值
    pvA_power288 = myinterpolation(pvA_power, '288')
    windA_power288 = myinterpolation(windA_power, '288')
    pvA_power96 = myinterpolation(pvA_power, '96')
    windA_power96 = myinterpolation(windA_power, '96')

    if output_amount == 96:
        loadA_Out = [loadA_total96, loadA_park96, loadA_pile96, loadA_lude96, loadA_zlzm96]
        powerA_Out = [pvA_power96, windA_power96]
    elif output_amount == 24:
        loadA_Out = [loadA_total, loadA_park, loadA_pile, loadA_lude, loadA_zlzm]
        powerA_Out = [pvA_power, windA_power]
    else:
        loadA_Out = [loadA_total288, loadA_park288, loadA_pile288, loadA_lude288, loadA_zlzm288]
        powerA_Out = [pvA_power288, windA_power288]
    loadA_Input = loadA_total96
    pvA_Input = pvA_power96
    windA_Input = windA_power96
    powerA_Input = myplusfunc([pvA_Input, windA_Input])

    loadB_total = myplusfunc([loadB_park, loadB_pile, loadB_lude, loadB_zlzm])
    # B区负荷预测结果，分别为园区负荷、充电桩、智慧路灯、直流照明；插值
    loadB_park288 = myinterpolation(loadB_park, '288')
    loadB_pile288 = myinterpolation(loadB_pile, '288')
    loadB_lude288 = myinterpolation(loadB_lude, '288')
    loadB_zlzm288 = myinterpolation(loadB_zlzm, '288')
    loadB_total288 = myinterpolation(loadB_total, '288')
    loadB_park96 = myinterpolation(loadB_park, '96')
    loadB_pile96 = myinterpolation(loadB_pile, '96')
    loadB_lude96 = myinterpolation(loadB_lude, '96')
    loadB_zlzm96 = myinterpolation(loadB_zlzm, '96')
    loadB_total96 = myinterpolation(loadB_total, '96')
    # B区新能源出力预测，分别为光伏、风机；插值
    pvB_power288 = myinterpolation(pvB_power, '288')
    pvB_power96 = myinterpolation(pvB_power, '96')

    if output_amount == 96:
        loadB_Out = [loadB_total96, loadB_park96, loadB_pile96, loadB_lude96, loadB_zlzm96]
        powerB_Out = pvB_power96
    elif output_amount == 24:
        loadB_Out = [loadB_total, loadB_park, loadB_pile, loadB_lude, loadB_zlzm]
        powerB_Out = pvB_power
    else:
        loadB_Out = [loadB_total288, loadB_park288, loadB_pile288, loadB_lude288, loadB_zlzm288]
        powerB_Out = pvB_power288
    loadB_Input = loadB_total96
    pvB_Input = pvB_power96
    [SOC1, SOC2] = getsoc() #这个是储能的当前容量吗？
    # print('数据预处理用时=', time.time()-begintime2)
    logger.info('开始优化求解')
    modelparas = model_define()
    model_amount = modelparas[1]
    for modelIndex in range(model_amount):
        begintime = time.time()
        optTarget = modelparas[0][modelIndex][0]
        optObject = modelparas[0][modelIndex][1]
        optTime = modelparas[0][modelIndex][2]
        optResult_id = modelparas[0][modelIndex][3][0]
        optName = modelparas[0][modelIndex][4]
        # print('运行优化模型参数加载用时=', time.time() - begintime)
        if optTarget == '' or optObject == '':
            logger.info('该模型未定义求解参数，跳过')
            continue
        else:
            if optObject == '1#储能':
                st = time.time()
                optInputA = [loadA_Input, powerA_Input, SOC1 / 100]
                optOut_strategyA, optOut_profitA = optimize1(optInputA, optTarget, output_amount)
                print('优化计算完成，用时=', time.time() - st)
                st = time.time()
                print('存储结果中···')
                try:
                    post_optresult_A(loadA_Out, powerA_Out, optOut_strategyA, optOut_profitA,
                                     optResult_id, output_amount)
                except:
                    raise ConnectionError
                else:
                    pass
                print('存储完成，用时=', time.time() - st)
            if optObject == '2#储能':
                st = time.time()
                optInputB = [loadB_Input, pvB_power96, SOC2 / 100]
                optOut_strategyB, optOut_profitB = optimize2(optInputB, optTarget, output_amount)
                print('优化计算完成，用时=', time.time() - st)
                st = time.time()
                print('存储结果中···')
                try:
                    post_optresult_B(loadB_Out, powerB_Out, optOut_strategyB, optOut_profitB,
                                     optResult_id, output_amount)
                except:
                    raise ConnectionError
                else:
                    pass
                print('存储完成，用时=', time.time() - st)
            if optObject == '全选':
                st = time.time()
                optInputA = [loadA_Input, powerA_Input, SOC1 / 100]
                optOut_strategyA, optOut_profitA = optimize1(optInputA, optTarget, output_amount)
                optInputB = [loadB_Input, pvB_power96, SOC2 / 100]
                optOut_strategyB, optOut_profitB = optimize2(optInputB, optTarget, output_amount)
                print('优化计算完成，用时=', time.time() - st)
                st = time.time()
                print('存储结果中···')
                try:
                    post_optresult_A(loadA_Out, powerA_Out, optOut_strategyA, optOut_profitA,
                                     optResult_id, output_amount)
                    post_optresult_B(loadB_Out, powerB_Out, optOut_strategyB, optOut_profitB,
                                     optResult_id, output_amount)
                except:
                    raise ConnectionError
                else:
                    pass
                print('存储完成，用时=', time.time() - st)
            logger.info('本模型求解完成')
            print('***', optName, '模型编号', modelIndex + 1, '计算用时=', time.time() - begintime)
    logger.info('完成所有模型求解，本轮计算结束')
