# 储能控制器主程序
# 调用配置文件与接口程序
# version0909 by zk
import datetime

from apscheduler.schedulers.background import BackgroundScheduler
import myinterface
from datetime import date
import time
import json
import requests
import numpy as np
import optimization_main
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
from utils import get_sim_soc_stra, env_sim

run_mode = 'remot'
if run_mode == 'remot':
    url0 = 'http://123.60.30.122:20930'
else:
    url0 = 'http://127.0.0.1:21530'
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
filename = log_path + delimeter + 'escontroler'
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

# 算法遥测历史数据-get接口
tsdb_user_get_ai = 'tsdb_user_get_ai_by_eqmid_and_id'
# 算法遥信历史数据-get接口
tsdb_user_get_di = 'tsdb_user_get_di_by_eqm_and_id'
# 算法实时数据-get接口
rtdb_user_get_all = 'rtdb_user_get_points_by_eqmid'
# 算法遥测历史数据-set接口
tsdb_user_set_ai = 'tsdb_user_set_ai_by_eqmid'
# 算法遥信历史数据-set接口
tsdb_user_set_di = 'tsdb_user_set_di_by_eqmid'
# 算法实时数据-set接口
rtdb_user_set_all = 'rtdb_user_set_points_by_eqmid'
# 采集遥测历史数据-get接口
tsdb_core_get_ai = 'tsdb_core_get_ai_by_eqm_and_id'
# 采集遥信历史数据-get接口
tsdb_core_get_di = 'tsdb_core_get_di_by_eqm_and_id'
# 采集实时数据-get接口
rtdb_core_get_all = 'rtdb_core_get_points_by_eqmid'

preid_list = list()
for i in range(8):
    # 预测总负荷、预测光伏、预测风机、预测储能（1#）
    str1 = 'opt0' + str(10001 + 10000 * i)
    str2 = 'opt0' + str(10011 + 10000 * i)
    str3 = 'opt0' + str(10019 + 10000 * i)
    str4 = 'opt0' + str(10021 + 10000 * i)
    # 预测总负荷、预测光伏、预测储能（2#）
    str5 = 'opt0' + str(10006 + 10000 * i)
    str6 = 'opt0' + str(10012 + 10000 * i)
    str7 = 'opt0' + str(10025 + 10000 * i)
    str8 = 'ctr0' + str(10001 + 10000 * i)
    str9 = 'ctr0' + str(10002 + 10000 * i)
    preid_list.append([str1, str2, str3, str4, str5, str6, str7])
outid_list = list()
for i in range(8):
    str1 = 'ctr0' + str(10001 + 10000 * i)
    str2 = 'ctr0' + str(10002 + 10000 * i)
    outid_list.append([str1, str2])

# 当前时刻
now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
now_minute = int(now[-5:-3])
now_sec = int(now[-2:])
if now_minute < 15:
    my_end = now[:-5] + '00:00'
elif 15 < now_minute < 30:
    my_end = now[:-5] + '15:00'
elif 30 < now_minute < 45:
    my_end = now[:-5] + '30:00'
elif now_minute > 45:
    my_end = now[:-5] + '45:00'
else:
    my_end = now

# 当天
today = date.today().strftime("%Y-%m-%d")
# 处理时间标记
hourpool = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
minutepool = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
minutepool2 = ['00', '15', '30', '45']

my_end_inds = 0
count = 0
for i_h in range(0, len(hourpool)):
    for j_m in range(0, len(minutepool2)):
        if my_end[-8:-6] == hourpool[i_h] and my_end[-5:-3] == minutepool2[j_m]:
            my_end_inds = count
        else:
            count += 1
my_end_inds -= 1
# print('my_end_inds', my_end_inds)

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


#  获取日前预测曲线，输出为[[],[],[],[],[],[],[]]
def get_today_pre(modelnum):
    begin0 = today + ' 00:00:00'
    end0 = today + ' 23:59:59'
    dura0 = '15m'
    eqm0 = 'eqm000000'
    idlist0 = preid_list[modelnum]
    datas = myinterface.get_tsdb_user_ai(begin0, end0, dura0, eqm0, idlist0)
    return datas


def get_soc_points():
    idlist0 = ['eqm007001:ai:BMS000008', 'eqm007002:ai:BMS000008']
    datas = myinterface.get_rtdb_core(idlist0)
    soc1 = datas[0]
    soc2 = datas[1]
    return soc1, soc2


def get_today_real():
    begin0 = today + ' 00:00:00'
    end0 = my_end
    dura0 = '15m'
    # 储能1出力
    eqm1 = 'eqm002004'
    idlist1 = ['DCM000003']
    # 储能2出力
    eqm2 = 'eqm002011'
    idlist2 = ['DCM000003']
    # 1区总负荷/光伏总出力/风机总出力,2区总负荷/光伏总出力
    eqm3 = 'eqm000000'
    idlist3 = ['frm000098', 'frm000099', 'frm000100', 'frm000102', 'frm000103']
    eqm4 = 'eqm007001'
    idlist4 = ['BMS000008']
    eqm5 = 'eqm007002'
    idlist5 = ['BMS000008']

    datas1 = myinterface.get_tsdb_core_ai(begin0, end0, dura0, eqm1, idlist1)
    datas2 = myinterface.get_tsdb_core_ai(begin0, end0, dura0, eqm2, idlist2)
    datas3 = myinterface.get_tsdb_user_ai(begin0, end0, dura0, eqm3, idlist3)
    datas4 = myinterface.get_tsdb_core_ai(begin0, end0, dura0, eqm4, idlist4)
    datas5 = myinterface.get_tsdb_core_ai(begin0, end0, dura0, eqm5, idlist5)
    for i in range(0, len(datas4[0])):
        if datas4[0][i] > 100:
            datas4[0][i] = 100
        if datas5[0][i] > 100:
            datas5[0][i] = 100
    # print('datas3', datas3)
    pne1_real = list()
    for i in range(0, len(datas3[1])):
        pne1_real.append(datas3[1][i] + datas3[2][i])
    return [datas1[0], datas3[0], pne1_real, datas2[0], datas3[3], datas3[-1], datas4[0], datas5[0]]


def get_dianliang():
    id_shoudian = 'frm000014'
    id_goudian = 'frm000018'
    eqm0 = 'eqm000000'
    begin0 = today + ' 00:00:00'
    end0 = my_end
    dura0 = '15m'
    datas = myinterface.get_tsdb_user_ai(begin0, end0, dura0, eqm0, [id_shoudian, id_goudian])
    if datas:
        return datas[0], datas[1]
    else:
        return [], []


# datalist1,datalist2:1#/2#储能优化控制策略，list类型，长度为96
def post_result(datalist1, datalist2):
    eqmnum0 = 'eqm000000'
    idlist0 = outid_list[0]
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }

    for inds in range(len(datalist1)):
        payload = {}
        key1 = eqmnum0 + ':ai:' + idlist0[0]
        key2 = eqmnum0 + ':ai:' + idlist0[1]
        payload[key1] = {'v': str(datalist1[inds]),
                         't': timestamp_list96[inds]}
        payload[key2] = {'v': str(datalist2[inds]),
                         't': timestamp_list96[inds]}
        payload = json.dumps(payload)
        requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payload, headers=headers)
        requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)


# 定义决策函数
def controller_optimization(x, y, Pre_ne, Pre_load, SOC, Eres, c1, Pres, Pch, Pdis, Pch_re, Pdis_re, Psell, Pbuy, i, B):
    P_Be = 0
    P_out = 0
    if abs(x[i] - y[i]) > 0.2:  # 1
        if Pre_ne[i] - Pre_load[i] > 0:  # 1.1、新能源出力大于负荷
            if SOC[i] < 0.9:  # 1.1.1、余电全部为储能充电
                if Eres * (0.9 - SOC[i]) < (Pre_ne[i] - Pre_load[i]) / 4:  # 1.1.1.1余电可以充满储能
                    Pch_re[i] = Eres * (0.9 - SOC[i]) * 4
                    if Pch_re[i] > Pres:  # 充电超过上限
                        Pch_re[i] = Pres
                    if Pch_re[i] < B * Pres:  # 充电低于下限
                        Pch_re[i] = 0
                    P_Be = - Pch_re[i]  # 储能放电策略
                    Psell[i] = Pre_ne[i] - Pre_load[i] - Pch_re[i]
                    P_out = - Psell[i]  # 电网购售电策略

                if Eres * (0.9 - SOC[i]) > (Pre_ne[i] - Pre_load[i]) / 4:  # 1.1.1.2余电未能充满储能
                    if c1[i] != min(c1):  # 电价为高峰、平段
                        Pch_1 = Pre_ne[i] - Pre_load[i]  # 一次充电
                        Pch_2 = 0  # 二次充电
                        Pch_re[i] = Pch_1 + Pch_2
                        if Pch_re[i] > Pres:
                            Pch_re[i] = Pres
                        if Pch_re[i] < B * Pres:
                            Pch_re[i] = 0
                        Psell[i] = Pre_ne[i] - Pre_load[i] - Pch_re[i]
                        Pbuy[i] = 0
                        P_Be = - Pch_re[i]  # 储能放电策略
                        P_out = Pbuy[i] - Psell[i]  # 电网购售电策略

                    if c1[i] == min(c1):  # 电价为低谷段
                        if Pch_re[i] * (1 + x[i] - y[i]) - (Pre_ne[i] - Pre_load[i]) > 0:  # 储能预测充电量大于新能源为储能充电量
                            Pch_1 = Pre_ne[i] - Pre_load[i]
                            Pch_2 = Pch_re[i] * (1 + x[i] - y[i]) - Pch_1
                            Pch_re[i] = Pch_1 + Pch_2
                            if Pch_re[i] > Pres:
                                Pch_re[i] = Pres
                                Psell[i] = Pre_ne[i] - Pre_load[i] - Pch_re[i]
                                Pbuy[i] = 0
                            if Pch_re[i] < B * Pres:
                                Pch_re[i] = 0
                                Psell[i] = 0
                                Pbuy[i] = Pre_ne[i] - Pre_load[i]
                        if Pch_re[i] * (1 + x[i] - y[i]) - (Pre_ne[i] - Pre_load[i]) < 0:  # 储能预测充电量小于新能源为储能充电量
                            Pch_1 = Pch_re[i] * (1 + x[i] - y[i])
                            Pch_2 = 0
                            if Pch_re[i] > Pres:
                                Pch_re[i] = Pres
                            if Pch_re[i] < B * Pres:
                                Pch_re[i] = 0
                            Psell[i] = Pre_ne[i] - Pre_load[i] - Pch_re[i]
                            Pbuy[i] = 0
                            P_Be = - Pch_re[i]  # 储能放电策略
                            P_out = Pbuy[i] - Psell[i]  # 电网购售电策略

            if SOC[i] >= 0.9:  # 1.1.2 余电全部上网
                Psell[i] = Pre_ne[i] - Pre_load[i]
                P_Be = Pdis_re[i] = Pch_re[i] = 0  # 储能策略
                P_out = -Psell[i]  # 电网策略

        if Pre_ne[i] - Pre_load[i] < 0:  # 1.2、新能源出力小于负荷
            if c1[i] != min(c1):  # 1.2.1 电价为高价、平段
                if SOC[i] > 0.2:  # 1.2.1.1
                    if Eres * (SOC[i] - 0.2) >= (Pre_load[i] - Pre_ne[i]) / 4:  # 储能放电可以满足剩余负荷需求
                        P_Be = Pre_load[i] - Pre_ne[i]
                        P_out = 0
                    else:  # 储能放电不能满足剩余负荷需求
                        Pdis_re[i] = Eres * (SOC[i] - 0.2) * 4
                        P_Be = Pdis_re[i]  # 储能策略
                        if Pch_re[i] > Pres:  # 充电超过上限
                            Pch_re[i] = Pres
                        if Pch_re[i] < B * Pres:  # 充电低于下限
                            Pch_re[i] = 0
                        Pbuy[i] = Pre_load[i] - Pre_ne[i] - P_Be
                        P_out = Pbuy[i]  # 电网策略
                if SOC[i] <= 0.2:  # 1.2.1.2
                    Pbuy[i] = Pre_load[i] - Pre_ne[i]
                    P_Be = 0
                    P_out = Pbuy[i]

            if c1[i] == min(c1):  # 1.2.2 电价为低谷段
                Pbuy_1 = Pre_load[i] - Pre_ne[i]
                if SOC[i] < 0.9:
                    if Pch_re[i] > 0:
                        Pch_re[i] = Pch_re[i] * (1 + y[i] - x[i])
                        Pbuy_2 = Pch_re[i]
                        Pbuy[i] = Pbuy_1 + Pbuy_2
                        P_Be = -Pch_re[i]
                        P_out = Pbuy[i]
                    else:
                        P_Be = Pdis_re[i] = Pch_re[i] = 0
                        Pbuy[i] = Pbuy_1
                        P_out = Pbuy[i]
    else:
        Pch_re[i] = Pch[i]
        Pdis_re[i] = Pdis[i]
        P_Be = Pdis_re[i] - Pch_re[i]
        Pbuy[i] = Pre_load[i] + Pch_re[i] + Psell[i] - Pre_ne[i] - Pdis_re[i]
        P_out = Pbuy[i] - Psell[i]
    return (P_Be, P_out)


def cal_a(Pload, Pne, Pch, Pdis, Pre_load, Pre_ne, SOC, Pch_re, Pdis_re, Psell, Pbuy, c1, Pres, Eres, Pes, B,
          opttarget, optinput):
    # 计数判断
    num = len(Pre_load)  # num时间点个数
    x = [0]*num  # 新能源出力预测偏差判断因子
    y = [0]*num   # 负荷预测偏差判断因子
    for i in range(0, num):
        x[i] = (Pre_ne[i] - Pne[i]) / (0.001+np.average(Pne[0:96]))
        y[i] = (Pre_load[i] - Pload[i]) / (0.001+np.average(Pload[0:96]))
    # 计数标识
    # 得到N_1和N_2值 最小值为重新优化的计数点
    count_1 = 0  # 计数标识1
    count_2 = 0  # 计数标识2
    for i in range(0, num):
        if abs(x[i]) > 0.5 or abs(y[i]) > 0.5:
            count_1 += 1
            if count_1 == 20:
                N_1 = i - 1
        if count_1 < 20:
            N_1 = num + 1
        if abs(x[i]) > 0.2 or abs(y[i]) > 0.2:
            count_2 += 1
            if count_2 == 32:
                N_2 = i - 1
        if count_2 < 32:
            N_2 = num + 1

    # 当计数标识达到第一次临界值后，重新开始计数
    if min(count_1, count_2) >= min(N_1, N_2):
        count_1_re = 0  # 计数标识1
        count_2_re = 0  # 计数标识1
        for i in range(min(N_1, N_2), num):
            if abs(x[i]) > 0.5 or abs(y[i]) > 0.5:
                count_1_re += 1
                if count_1_re == 20:
                    N_1_re = i - 1
            if count_1_re < 20:
                N_1_re = num + 1
            if abs(x[i]) > 0.2 or abs(y[i]) > 0.2:
                count_2_re += 1
                if count_2_re == 32:
                    N_2_re = i - 1
            if count_2_re < 32:
                N_2_re = num + 1
    # 情况1
    if min(count_1, count_2) < min(N_1, N_2):  # 当时间点数小于重新优化临街值时，执行优化策略
        for i in range(0, num):
            P_Be = np.empty([96])
            P_out = np.empty([num])
            [P_Be_1, P_out_1] = controller_optimization(x, y, Pre_ne, Pre_load, SOC, Eres, c1, Pres, Pch_re,
                                                        Pdis_re, Pch, Pdis,  Psell, Pbuy, i, B)  # 执行控制器优化策略
            P_Be[i] = P_Be_1
            P_out[i] = P_out_1
        P_Be[num:96] = Pes[num:96]

    if min(count_1, count_2) >= min(N_1, N_2):  # 当时间点数大于等于临界值，重新优化后执行优化策略
        P_Be = [0]*96
        P_out = [0]*num
        for i in range(0, min(N_1, N_2)):  # 对于时间点数小于临界值的点数 ，执行优化策略
            [P_Be_1, P_out_1] = controller_optimization(x, y, Pre_ne, Pre_load, SOC, Eres, c1, Pres, Pch_re,
                                                        Pdis_re, Pch, Pdis,  Psell, Pbuy, i, B)  # 执行控制器优化策略
            P_Be[i] = P_Be_1
            P_out[i] = P_out_1
        # 当时间点数到达临界值，重新预测当前时间点后24小时的预测曲线，进行重新优化（当前选择成本最优为目标进行优化）
        # ---------------------------------
        # 新的输入参数定义
        # (Pload, Pne, Pch, Pdis, Pre_load, Pre_ne, SOC, Pch_re, Pdis_re, Psell, Pbuy, c1, Pres, Eres, Pes, B)
        # 需重新定义输入：Pload（num）,Pne（num） ,c1（96）  重新拼接96          其余外部数据输入：c2,B,SOC0,Pres,Eres
        # Pload_re =
        # Pne_re =
        c1_re = np.zeros([96])  # 电价
        c1_re[0:96 - num] = c1[num:96]
        c1_re[96 - num:96] = c1[0:num]

        Pes1, optOut_profitA = optimization_main.optimize1(optinput, opttarget, 96)
        print('Pes1', len(Pes1))

        Pch1 = np.zeros([96])
        Pdis1 = np.zeros([96])
        for i in range(0, len(Pes1)):
            if Pes1[i] >= 0:
                Pdis1[i] = Pes1[i]
            else:
                Pch1[i] = Pes1[i]
        Pload1 = optinput[0]
        Pne1 = optinput[1]

        # ---------------------------------
        # #输出重新优化后当日的一个运行优化曲线。
        # P_Be_re = np.empty([0,96])
        # P_out_re = np.empty([0, 96])
        # P_Be_re[0:min(N_1, N_2)] = P_Be[0:min(N_1, N_2)]
        # P_Be_re[min(N_1, N_2):96] =P_Be_re[0:96-min(N_1, N_2)]
        # P_out_re[0:min(N_1, N_2)] = P_out[0:min(N_1, N_2)]
        # P_out_re[min(N_1, N_2):96] = P_out_re[0:96-min(N_1, N_2)]
        # 重新计算临界值及后续节点
        for i in range(min(N_1, N_2), num):
            [P_Be_opt, P_out_opt] = controller_optimization(x, y, Pne1, Pload1, SOC, Eres, c1, Pres, Pch_re, Pdis_re,
                                                            Pch1, Pdis1, Psell, Pbuy, i, B)
            P_Be[i] = P_Be_opt
            P_out[i] = P_out_opt
        P_Be[num:96] = Pes1[0:96 - num]
    return P_Be, P_out


def cal_b(Pload, Pne, Pch, Pdis, Pre_load, Pre_ne, SOC, Pch_re, Pdis_re, Psell, Pbuy, c1, Pres, Eres, Pes, B,
          opttarget, optinput):
    # 计数判断
    num = len(Pre_load)  # num时间点个数
    x = np.empty([num])  # 新能源出力预测偏差判断因子
    y = np.empty([num])  # 负荷预测偏差判断因子
    for i in range(0, num):
        x[i] = (Pre_ne[i] - Pne[i]) / np.average(Pne[0:96])
        y[i] = (Pre_load[i] - Pload[i]) / np.average(Pload[0:96])
    # 计数标识
    # 得到N_1和N_2值 最小值为重新优化的计数点
    count_1 = 0  # 计数标识1
    count_2 = 0  # 计数标识2
    for i in range(0, num):
        if abs(x[i]) > 0.5 or abs(y[i]) > 0.5:
            count_1 += 1
            if count_1 == 20:
                N_1 = i - 1
        if count_1 < 20:
            N_1 = num + 1
        if abs(x[i]) > 0.2 or abs(y[i]) > 0.2:
            count_2 += 1
            if count_2 == 32:
                N_2 = i - 1
        if count_2 < 32:
            N_2 = num + 1

    # 当计数标识达到第一次临界值后，重新开始计数
    if min(count_1, count_2) >= min(N_1, N_2):
        count_1_re = 0  # 计数标识1
        count_2_re = 0  # 计数标识1
        for i in range(min(N_1, N_2), num):
            if abs(x[i]) > 0.5 or abs(y[i]) > 0.5:
                count_1_re += 1
                if count_1_re == 20:
                    N_1_re = i - 1
            if count_1_re < 20:
                N_1_re = num + 1
            if abs(x[i]) > 0.2 or abs(y[i]) > 0.2:
                count_2_re += 1
                if count_2_re == 32:
                    N_2_re = i - 1
            if count_2_re < 32:
                N_2_re = num + 1
    # 情况1
    if min(count_1, count_2) < min(N_1, N_2):  # 当时间点数小于重新优化临街值时，执行优化策略
        for i in range(0, num):
            P_Be = np.empty([96])
            P_out = np.empty([num])
            [P_Be_1, P_out_1] = controller_optimization(x, y, Pre_ne, Pre_load, SOC, Eres, c1, Pres, Pch, Pdis, Pch_re,
                                                        Pdis_re, Psell, Pbuy, i, B)  # 执行控制器优化策略
            P_Be[i] = P_Be_1
            P_out[i] = P_out_1
        P_Be[num:96] = Pes[num:96]

    if min(count_1, count_2) >= min(N_1, N_2):  # 当时间点数大于等于临界值，重新优化后执行优化策略
        P_Be = np.empty([96])
        P_out = np.empty([num])
        for i in range(0, min(N_1, N_2)):  # 对于时间点数小于临界值的点数 ，执行优化策略
            [P_Be_1, P_out_1] = controller_optimization(x, y, Pre_ne, Pre_load, SOC, Eres, c1, Pres, Pch, Pdis, Pch_re,
                                                        Pdis_re, Psell, Pbuy, i, B)  # 执行控制器优化策略
            P_Be[i] = P_Be_1
            P_out[i] = P_out_1
        # 当时间点数到达临界值，重新预测当前时间点后24小时的预测曲线，进行重新优化（当前选择成本最优为目标进行优化）
        # ---------------------------------
        # 新的输入参数定义
        # (Pload, Pne, Pch, Pdis, Pre_load, Pre_ne, SOC, Pch_re, Pdis_re, Psell, Pbuy, c1, Pres, Eres, Pes, B)
        # 需重新定义输入：Pload（num）,Pne（num） ,c1（96）  重新拼接96          其余外部数据输入：c2,B,SOC0,Pres,Eres
        # Pload_re =
        # Pne_re =
        c1_re = np.zeros([96])  # 电价
        c1_re[0:96 - num] = c1[num:96]
        c1_re[96 - num:96] = c1[0:num]

        Pes1, optOut_profit = optimization_main.optimize2(optinput, opttarget, 96)
        Pch1 = np.zeros([96])
        Pdis1 = np.zeros([96])
        for i in range(0, len(Pes1)):
            if Pes1[i] >= 0:
                Pdis1[i] = Pes1[i]
            else:
                Pch1[i] = Pes1[i]
        Pload1 = optinput[0]
        Pne1 = optinput[1]

        # ---------------------------------
        # #输出重新优化后当日的一个运行优化曲线。
        # P_Be_re = np.empty([0,96])
        # P_out_re = np.empty([0, 96])
        # P_Be_re[0:min(N_1, N_2)] = P_Be[0:min(N_1, N_2)]
        # P_Be_re[min(N_1, N_2):96] =P_Be_re[0:96-min(N_1, N_2)]
        # P_out_re[0:min(N_1, N_2)] = P_out[0:min(N_1, N_2)]
        # P_out_re[min(N_1, N_2):96] = P_out_re[0:96-min(N_1, N_2)]
        # 重新计算临界值及后续节点
        for i in range(min(N_1, N_2), num):
            [P_Be_opt, P_out_opt] = controller_optimization(x, y, Pne1, Pload1, SOC, Eres, c1, Pres, Pch1, Pdis1,
                                                            Pch_re, Pdis_re, Psell, Pbuy, i, B)
            P_Be[i] = P_Be_opt
            P_out[i] = P_out_opt
        P_Be[num:96] = Pes1[0:96 - num]
    return P_Be, P_out


def post_ctrresult(data_timelist, userid):
    # print(data_timelist)
    eqmnum = 'eqm000000'
    mytoday = date.today().strftime("%Y-%m-%d")
    stamplist = timestamp_list96

    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    for j in range(0, len(stamplist)):
        payload = {}
        key = eqmnum + ':ai:' + userid
        payload[key] = {'v': str(data_timelist[j]),
                        't': mytoday + ' ' + stamplist[j]}
        payload = json.dumps(payload)

        requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payload, headers=headers)
        requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)
        # print('post done, timenum=', j)


def post_ctrresult_all(data_timelist1, data_timelist2, userid1, userid2):
    # print(data_timelist1)
    # print(data_timelist2)
    eqmnum = 'eqm000000'
    mytoday = date.today().strftime("%Y-%m-%d")
    stamplist = timestamp_list96

    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    for j in range(0, len(stamplist)):
        payload = {}
        key1 = eqmnum + ':ai:' + userid1
        payload[key1] = {'v': str(data_timelist1[j]),
                         't': mytoday + ' ' + stamplist[j]}
        key2 = eqmnum + ':ai:' + userid2
        payload[key2] = {'v': str(data_timelist2[j]),
                         't': mytoday + ' ' + stamplist[j]}
        payload = json.dumps(payload)

        requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payload, headers=headers)
        requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)


def myfunc():
    logger.info('本轮计算开始')
    sttime = time.time()
    # 获取真实数据
    [p_past_es1, p_past_load1, p_past_ne1, p_past_es2, p_past_load2, p_past_ne2, soc1_past,
     soc2_past] = get_today_real()
    # pastdata = [p_past_es1, p_past_load1, p_past_ne1, p_past_es2, p_past_load2, p_past_ne2, soc1_past, soc2_past]
    # for i in range(0, len(pastdata)):
    #     print(i, len(pastdata[i]))
    modelparas = optimization_main.model_define()
    model_amount = modelparas[1]
    # SOC1 = soc1_past
    # SOC2 = soc2_past
    p_past_es1, SOC1 = get_sim_soc_stra(datetime.datetime.now())
    p_past_es2, SOC2 = get_sim_soc_stra(datetime.datetime.now())
    # print(model_amount)
    for modelIndex in range(0, model_amount):
        print('modelnum=', modelIndex)
        begintime = time.time()
        optTarget = modelparas[0][modelIndex][0]
        # print(optTarget)
        optObject = modelparas[0][modelIndex][1]
        # print(optObject)
        optTime = modelparas[0][modelIndex][2]
        optResult_id = modelparas[0][modelIndex][3][0]
        optName = modelparas[0][modelIndex][4]
        # print('运行优化模型参数加载用时=', time.time() - begintime)
        if optTarget == '' or optObject == '':
            continue
        else:
            '''获取该模型下的输入量'''
            # 外部输入量
            c1 = np.array([0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784, 0.2784,
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
            Pres = 250
            Eres = 500
            B = 0.1
            # '''# 1#、2#储能SOC实时值'''
            # SOC1, SOC2 = get_soc_points()
            # if SOC1 > 100:
            #     SOC1 = 100
            # if SOC2 > 100:
            #     SOC2 = 100

            if optObject == '1#储能':
                '''预测数据'''
                # 获取预测数据
                predatas = get_today_pre(modelIndex)
                # 以下分别为1区预测负荷、光伏出力、风机出力、储能功率（正为放电、负为充电），均为list类型，长度96
                pre_load1 = predatas[0]
                pre_pv1 = predatas[1]
                pre_wind1 = predatas[2]
                pre_es1 = predatas[3]
                pre_ne1 = optimization_main.myplusfunc([pre_pv1, pre_wind1])

                '''# 分区1输入'''
                Pload_a = pre_load1
                Pne_a = np.array(pre_pv1) + np.array(pre_wind1)
                Pes_a = pre_es1
                Pch_a = np.zeros([len(p_past_es1)])
                Pdis_a = np.zeros([len(p_past_es1)])
                for i in range(0, len(p_past_es1)):
                    if p_past_es1[i] >= 0:
                        Pdis_a[i] = p_past_es1[i]
                    else:
                        Pch_a[i] = p_past_es1[i]
                # -------------------------------已调整----------------------------------------
                Pre_load_a = np.array(p_past_load1)  # 实际负荷出力数据--采集输入
                Pre_ne_a = np.array(p_past_ne1)  # 实际新能源出力数据--采集输入

                e_past_sell, e_past_buy = get_dianliang()
                Psell_a = np.array(e_past_sell)
                Pbuy_a = np.array(e_past_buy)
                # -------------------------------已调整----------------------------------------
                Pch_re_a = np.zeros([len(p_past_es1)])
                Pdis_re_a = np.zeros([len(p_past_es1)])
                for i in range(0, len(p_past_es1)):
                    if p_past_es1[i] >= 0:
                        Pch_re_a[i] = p_past_es1[i]
                    else:
                        Pdis_re_a[i] = -p_past_es1[i]
                SOC_a = SOC1

                loadA_Input = list()
                powerA_Input = list()
                for inds in range(0, 96):
                    if inds <= my_end_inds:
                        loadA_Input.append(p_past_load1[inds])
                        powerA_Input.append(p_past_ne1[inds])
                    else:
                        loadA_Input.append(pre_load1[inds])
                        powerA_Input.append(pre_ne1[inds])

                optInputA = [loadA_Input, powerA_Input, SOC_a[0] / 100]
                [P_Be_a, P_out_a] = cal_a(Pload_a, Pne_a, Pch_a, Pdis_a, Pre_load_a, Pre_ne_a, SOC_a, Pch_re_a,
                                          Pdis_re_a, Psell_a, Pbuy_a, c1, Pres, Eres, Pes_a, B, optTarget, optInputA)
                st = time.time()
                post_ctrresult(P_Be_a, outid_list[modelIndex][0])
                print('写数耗时', time.time() - st)
            if optObject == '2#储能':
                '''预测数据'''
                predatas = get_today_pre(modelIndex)
                # 以下分别为2区预测负荷、光伏出力、储能功率（正为放电、负为充电），均为list类型，长度96
                pre_load2 = predatas[4]
                pre_pv2 = predatas[5]
                pre_es2 = predatas[6]

                '''# 分区2输入'''
                Pload_b = pre_load2
                Pne_b = pre_pv2
                Pes_b = pre_es2
                Pch_b = np.zeros([len(p_past_es2)])
                Pdis_b = np.zeros([len(p_past_es2)])
                for i in range(0, len(p_past_es2)):
                    if p_past_es2[i] >= 0:
                        Pdis_b[i] = p_past_es2[i]
                    else:
                        Pch_b[i] = p_past_es2[i]
                # -------------------------------已调整----------------------------------------
                Pre_load_b = np.array(p_past_load2)  # 实际负荷出力数据--采集输入
                Pre_ne_b = np.array(p_past_ne2)  # 实际新能源出力数据--采集输入

                e_past_sell, e_past_buy = get_dianliang()
                Psell_b = np.array(e_past_sell)
                Pbuy_b = np.array(e_past_buy)
                # -------------------------------已调整----------------------------------------
                Pch_re_b = np.zeros([len(p_past_es2)])
                Pdis_re_b = np.zeros([len(p_past_es2)])
                for i in range(0, len(p_past_es1)):
                    if p_past_es1[i] >= 0:
                        Pch_re_b[i] = p_past_es2[i]
                    else:
                        Pdis_re_b[i] = -p_past_es2[i]
                SOC_b = SOC2

                loadB_Input = list()
                powerB_Input = list()
                for inds in range(0, 96):
                    if inds <= my_end_inds:
                        loadB_Input.append(p_past_load2[inds])
                        powerB_Input.append(p_past_ne2[inds])
                    else:
                        loadB_Input.append(pre_load2[inds])
                        powerB_Input.append(pre_pv2[inds])

                optInputB = [loadB_Input, powerB_Input, SOC_b[0] / 100]
                [P_Be_b, P_out_b] = cal_b(Pload_b, Pne_b, Pch_b, Pdis_b, Pre_load_b, Pre_ne_b, SOC_b, Pch_re_b,
                                          Pdis_re_b, Psell_b, Pbuy_b, c1, Pres, Eres, Pes_b, B,
                                          optTarget, optInputB)
                st = time.time()
                post_ctrresult(P_Be_b, outid_list[modelIndex][1])
                print('写数耗时', time.time() - st)
            if optObject == '全选':
                '''预测数据'''
                predatas = get_today_pre(modelIndex)
                # 以下分别为1区预测负荷、光伏出力、风机出力、储能功率（正为放电、负为充电），均为list类型，长度96
                pre_load1 = predatas[0] #166
                pre_pv1 = predatas[1] #0
                pre_wind1 = predatas[2] #0
                pre_es1 = predatas[3] #-250
                pre_ne1 = optimization_main.myplusfunc([pre_pv1, pre_wind1])
                # 以下分别为2区预测负荷、光伏出力、储能功率（正为放电、负为充电），均为list类型，长度96
                pre_load2 = predatas[4] #221
                pre_pv2 = predatas[5] #0
                pre_es2 = predatas[6] #250

                '''# 分区1输入'''
                Pload_a = pre_load1
                Pne_a = np.array(pre_pv1) + np.array(pre_wind1)
                Pes_a = pre_es1
                # 怀疑这个是预测值 pre_es1
                Pch_a = np.zeros([len(pre_es1)])
                Pdis_a = np.zeros([len(pre_es1)])
                for i in range(0, len(pre_es1)):
                    if pre_es1[i] >= 0:
                        Pdis_a[i] = pre_es1[i]
                    else:
                        Pch_a[i] = pre_es1[i]
                # -------------------------------已调整----------------------------------------
                Pre_load_a = np.array(p_past_load1)  # 实际负荷出力数据--采集输入
                Pre_ne_a = np.array(p_past_ne1)  # 实际新能源出力数据--采集输入

                e_past_sell, e_past_buy = get_dianliang()
                Psell_a = np.array(e_past_sell)
                Pbuy_a = np.array(e_past_buy)
                # -------------------------------已调整----------------------------------------
                Pch_re_a = np.zeros([len(p_past_es1)])
                Pdis_re_a = np.zeros([len(p_past_es1)])
                for i in range(0, len(p_past_es1)):
                    if p_past_es1[i] >= 0:
                        Pdis_re_a[i] = p_past_es1[i]
                    else:
                        Pch_re_a[i] = -p_past_es1[i]
                SOC_a = SOC1
                '''# 分区2输入'''
                Pload_b = pre_load2
                Pne_b = pre_pv2
                Pes_b = pre_es2
                Pch_b = np.zeros([len(pre_es2)])
                Pdis_b = np.zeros([len(pre_es2)])
                for i in range(0, len(pre_es2)):
                    if pre_es2[i] >= 0:
                        Pdis_b[i] = pre_es2[i]
                    else:
                        Pch_b[i] = pre_es2[i]
                # -------------------------------已调整----------------------------------------
                Pre_load_b = np.array(p_past_load2)  # 实际负荷出力数据--采集输入
                Pre_ne_b = np.array(p_past_ne2)  # 实际新能源出力数据--采集输入

                Psell_b = np.array(e_past_sell)
                Pbuy_b = np.array(e_past_buy)
                # -------------------------------已调整----------------------------------------
                Pch_re_b = np.zeros([len(p_past_es2)])
                Pdis_re_b = np.zeros([len(p_past_es2)])
                for i in range(0, len(p_past_es2)):
                    if p_past_es2[i] >= 0:
                        Pdis_re_b[i] = p_past_es2[i]
                    else:
                        Pch_re_b[i] = -p_past_es2[i]
                SOC_b = SOC2

                loadA_Input = list()
                powerA_Input = list()
                loadB_Input = list()
                powerB_Input = list()
                for inds in range(0, 96):
                    if inds <= my_end_inds:
                        loadA_Input.append(p_past_load1[inds])
                        powerA_Input.append(p_past_ne1[inds])
                        loadB_Input.append(p_past_load2[inds])
                        powerB_Input.append(p_past_ne2[inds])
                    else:
                        loadA_Input.append(pre_load1[inds])
                        powerA_Input.append(pre_ne1[inds])
                        loadB_Input.append(pre_load2[inds])
                        powerB_Input.append(pre_pv2[inds])
                # print('inputlength', len(loadA_Input), len(powerA_Input))
                optInputA = [loadA_Input, powerA_Input, SOC_a[0] / 100]
                tmp_soc_a = list(map(lambda x: x/100, SOC_a))
                [P_Be_a, P_out_a] = cal_a(Pload_a, Pne_a, Pch_a, Pdis_a, Pre_load_a, Pre_ne_a, tmp_soc_a, Pch_re_a, #pre pre real real
                                          Pdis_re_a, Psell_a, Pbuy_a, c1, Pres, Eres, Pes_a, B, optTarget,
                                          optInputA)
                optInputB = [loadB_Input, powerB_Input, SOC_b[0] / 100]
                tmp_soc_b = list(map(lambda x: x / 100, SOC_b))
                [P_Be_b, P_out_b] = cal_b(Pload_b, Pne_b, Pch_b, Pdis_b, Pre_load_b, Pre_ne_b, tmp_soc_b, Pch_re_b,
                                          Pdis_re_b, Psell_b, Pbuy_b, c1, Pres, Eres, Pes_b, B,
                                          optTarget, optInputB)
                st = time.time()
                post_ctrresult_all(P_Be_a, P_Be_b, outid_list[modelIndex][0], outid_list[modelIndex][1])
                print('写数耗时', time.time() - st)
        logger.info('完成一个模型计算')
    print('一轮耗时=', time.time() - sttime)
    logger.info('本轮计算结束')


if __name__ == '__main__':
    myfunc()
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(myfunc, trigger='cron', minute='*/15', second=0, timezone='Asia/Shanghai')
    # scheduler.start()
    # try:
    #     while True:
    #         time.sleep(2)
    # except SystemExit:
    #     scheduler.shutdown()
