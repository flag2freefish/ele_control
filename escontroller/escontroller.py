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
#TODO 碳排因子？
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

out_1_id_list = list()
'''
1#M1总负荷预测精准度
2#M1总负荷预测精准度
1#M1总负荷预测最大误差
2#M1总负荷预测最大误差
1#M1光伏出力预测精准度
2#M1光伏出力预测精准度
1#M1光伏出力预测最大误差
2#M1光伏出力预测最大误差
'''
for i in range(8):
    str1 = 'ctr0' + str(10019 + 10000 * i)
    str2 = 'ctr0' + str(10020 + 10000 * i)
    str3 = 'ctr0' + str(10021 + 10000 * i)
    str4 = 'ctr0' + str(10022 + 10000 * i)
    str5 = 'ctr0' + str(10023 + 10000 * i)
    str6 = 'ctr0' + str(10024 + 10000 * i)
    str7 = 'ctr0' + str(10025 + 10000 * i)
    str8 = 'ctr0' + str(10026 + 10000 * i)
    out_1_id_list.append([str1, str2, str3, str4, str5, str6, str7, str8])

out_2_id_list = list()
'''
1#储能控制器经济成本优化
2#储能控制器经济成本优化
1#储能控制器碳排成本优化
2#储能控制器碳排成本优化
1#M1储能控制器策略经济成本
2#M1储能控制器策略经济成本
1#M1储能控制器策略碳排成本
2#M1储能控制器策略碳排成本
'''
for i in range(8):
    str1 = 'ctr0' + str(10003 + 10000 * i)
    str2 = 'ctr0' + str(10004 + 10000 * i)
    str3 = 'ctr0' + str(10005 + 10000 * i)
    str4 = 'ctr0' + str(10006 + 10000 * i)
    str5 = 'ctr0' + str(10007 + 10000 * i)
    str6 = 'ctr0' + str(10008 + 10000 * i)
    str7 = 'ctr0' + str(10009 + 10000 * i)
    str8 = 'ctr0' + str(10010 + 10000 * i)
    out_2_id_list.append([str1, str2, str3, str4, str5, str6, str7, str8])

out_3_id_list = list()
'''
1#M1日前预测储能策略经济成本
2#M1日前预测储能策略经济成本
1#M1日前预测储能策略碳排成本
2#M1日前预测储能策略碳排成本
1#M1两充两放经济成本
2#M1两充两放经济成本
1#M1两充两放碳排成本
2#M1两充两放碳排成本
'''
for i in range(8):
    str1 = 'ctr0' + str(10011 + 10000 * i)
    str2 = 'ctr0' + str(10012 + 10000 * i)
    str3 = 'ctr0' + str(10013 + 10000 * i)
    str4 = 'ctr0' + str(10014 + 10000 * i)
    str5 = 'ctr0' + str(10015 + 10000 * i)
    str6 = 'ctr0' + str(10016 + 10000 * i)
    str7 = 'ctr0' + str(10017 + 10000 * i)
    str8 = 'ctr0' + str(10018 + 10000 * i)
    out_3_id_list.append([str1, str2, str3, str4, str5, str6, str7, str8])

# 处理时间标记
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


#  获取日前预测曲线，输出为[[],[],[],[],[],[],[]]
def get_today_pre(modelnum, input_time):
    begin0 = input_time.strftime("%Y-%m-%d") + ' 00:00:00'
    end0 = input_time.strftime("%Y-%m-%d") + ' 23:59:59'
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


def get_today_real(input_time, minute_):
    minute_now = input_time.minute // minute_ * minute_
    end0 = input_time.replace(minute=minute_now).strftime("%Y-%m-%d %H:%M:%S")
    begin0 = input_time.strftime("%Y-%m-%d") + ' 00:00:00'
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


def get_dianliang(input_time, minute_):
    #time_now = datetime.datetime.now()
    minute_now = input_time.minute // minute_ * minute_
    my_end = input_time.replace(minute=minute_now).strftime("%Y-%m-%d %H:%M:%S")
    id_shoudian = 'frm000014'
    id_goudian = 'frm000018'
    eqm0 = 'eqm000000'
    begin0 = input_time.strftime("%Y-%m-%d") + ' 00:00:00'
    dura0 = '15m'
    datas = myinterface.get_tsdb_user_ai(begin0, my_end, dura0, eqm0, [id_shoudian, id_goudian])
    if datas:
        return datas[0], datas[1]
    else:
        return [], []

def controller_optimization1(x, y, Pre_ne, Pre_load, SOC, Eres, Pres, c1, Pch, Pdis, i):
    '''
    策略控制器，计算当前时刻下，储能充放电量
    :param x: 负荷 预测值的偏差因子
    :param y: 新能源出力 预测值的偏差因子
    :param Pre_ne: 真实的新能源出力
    :param Pre_load: 真实的负荷
    :param SOC: 储能设备SOC
    :param Eres: 储能放电最大值
    :param Pres: 储能充电最大值
    :param Pch: 预测储能设备充电值
    :param Pdis: 预测储能设备放电值
    :param i: 第i时刻
    :return: 优化后第i时刻储能设备充放电策略
    '''
    # 当前偏差较大，通过该控制器重新计算策略
    if abs(x[i] - y[i]) > 0.2:
        if Pre_ne[i] >= Pre_load[i]:
            if SOC[i]<0.9:
                Pch_1 = Pre_ne[i] - Pre_load[i]
                if c1[i] == min(c1):
                    Pch_tmp = Pch[i] * (1 + x[i] - y[i]) - Pch_1
                    Pdis_tmp = 0
                else:
                    Pch_tmp = Pch_1
                    Pdis_tmp = 0
            else:
                Pch_tmp = 0
                Pdis_tmp = 0
        else:
            if c1[i] != min(c1):  # 1.2.1 电价为高价、平段
                if SOC[i] > 0.2:  # 1.2.1.1
                    # 储能放电可以满足剩余负荷需求
                    if Eres * (SOC[i] - 0.2) >= (Pre_load[i] - Pre_ne[i]) / 4:
                        Pdis_tmp = Pre_load[i] - Pre_ne[i]
                        Pch_tmp = 0
                    else:  # 储能放电不能满足剩余负荷需求
                        Pdis_tmp = Eres * (SOC[i] - 0.2) * 4
                        Pch_tmp = 0
                else:  # 1.2.1.2
                    Pch_tmp = 0
                    Pdis_tmp = 0
            else:  # 1.2.2 电价为低谷段
                if SOC[i] < 0.9 and Pch[i] > 0:
                    Pch_tmp = max(Pch[i] * (1 + y[i] - x[i]), Pres)
                    Pdis_tmp = 0
                else:
                    Pch_tmp = 0
                    Pdis_tmp = 0
        #充放电功率低于最大值
        return min(Pdis_tmp, Pres, Eres * (SOC[i]-min(0.2, SOC[i]) )*4) - min(Pch_tmp, Pres, Eres * (max(0.9, SOC[i]) - SOC[i])*4)
    else:
        return min(Pdis[i], Eres * (SOC[i]-min(0.2, SOC[i]) )*4) - min(Pch[i], Eres * (max(0.9, SOC[i]) - SOC[i])*4)

def update_soc(pb, soc, Eres):
    soc.append(soc[-1]+(-1*pb/4/Eres))
    return soc

def env_soc(count=4):
    a = [-250]*12*2 + [0]*12*8 + [250]*12*2 + [0]*12 + [-250] *12*2 + [0]*12*2 + [250] *12*2 +[0]*12*5

    b = list(map(lambda x: (x/24)*100+1, range(24))) + [99]*12*8 \
        + list(map(lambda x: (x/24)*100-1, range(24,0, -1))) + [1] * 12 \
        + list(map(lambda x: (x/24)*100+1, range(24))) + [99] * 12*2 \
        + list(map(lambda x: (x/24)*100-1, range(24,0, -1))) + [1] * 12*5
    tmp_div = int(12/count)
    return a[::tmp_div].copy(), b[::tmp_div].copy()

def cal_new_stra(Pload, Pne, Pch, Pdis, Pre_load, Pre_ne, SOC, Pch_re, Pdis_re, c1, Pres, Eres, Pes, B,
          opttarget, optinput):
    '''
    新策略条件判断
    :param Pload: 预测负荷
    :param Pne:  预测光伏出力
    :param Pch: 预测储能充电策略
    :param Pdis: 预测储能放电策略
    :param Pre_load: 真实负荷
    :param Pre_ne: 真实光伏出力
    :param SOC: 储能SOC
    :param Pch_re: 真实储能充电策略
    :param Pdis_re: 真实储能放电策略
    :param c1:  购电价格
    :param Pres: 储能充放电最大功率
    :param Eres: 储能设备最大容量
    :param Pes:  预测储能策略
    :param B:  储能放电最小参数
    :param opttarget:  优化目标：成本最优、碳排最优
    :param optinput:  重新计算优化任务下的输入
    :return:  新的策略
    '''
    # 计数判断
    num = len(Pre_load)  # num时间点个数 真实数据的长度
    print(num)
    if num == 0:
        return Pes, []
    if num == 60:
        print('Pass')
    x = [0]*num  # 新能源出力预测偏差判断因子, 设置最大值是1，由于数据出现异常情况，真实数据中晚上也会有光伏出力的情况
    y = [0]*num  # 负荷预测偏差判断因子, 设置最大值是1
    for i in range(0, num):
        x[i] = min((Pre_ne[i] - Pne[i]) / (0.01+np.average(Pne[0:96])), 1)
        x[i] = max(x[i], -1)
        y[i] = min((Pre_load[i] - Pload[i]) / (0.01+np.average(Pload[0:96])), 1)
        y[i] = max(y[i], -1)
    # 计数标识
    list_5 = list(map(lambda m, n: 1 if m > 0.5 or n > 0.5 else 0, x, y))
    list_2 = list(map(lambda m, n: 1 if m > 0.2 or n > 0.2 else 0, x, y))
    flag_opt = sum(list_5) >= 32 or sum(list_2) >= 20
    P_Be = -1 * Pch_re + Pdis_re
    # 首先取预测的策略作为当前策略，初始化值，会影响优化后的策略
    P_Be = np.concatenate([P_Be[:num-1], Pes[num-1:96]])
    # 当时间点数大于等于临界值，重新优化后执行优化策略，目前只是根据真实值拼接预测值，直接计算策略
    #TODO 需要加入重新预测流程
    if flag_opt:
        Pes1, optOut_profit = optimization_main.optimize2(optinput, opttarget, 96)
        # 当前优化点的策略重新赋值，是重新计算之后的策略
        P_Be[num-1:96] = Pes1[num-1:96]
        Pch_ = np.zeros([96])
        Pdis_ = np.zeros([96])
        for i in range(0, len(Pes1)):
            if Pes1[i] >= 0:
                Pdis_[i] = Pes1[i]
            else:
                Pch_[i] = -1*Pes1[i]
    else:
        Pch_ = Pch.copy()
        Pdis_ = Pdis.copy()
    # 执行控制器优化策略,对num-1位置的策略进行优化，当前位置，以15分钟为间隔， 0-15分钟是0分时刻， 15-30分钟是15分钟时刻，返回策略只有一个
    P_Be_1 = controller_optimization1(x, y, Pre_ne, Pre_load, SOC, Eres, Pres, c1, Pch_, Pdis_, num-1)
    # 将返回结果合并至策略中 TODO 放电策略修正
    if P_Be_1 < 0 and P_Be_1 > -1*B*Pres:
        P_Be[num - 1] = 0
    else:
        P_Be[num - 1] = P_Be_1
    return P_Be, []

# datalist1,datalist2:1#/2#储能优化控制策略，list类型，长度为96
def post_result(datalist1, datalist2):
    eqmnum0 = 'eqm000000'
    idlist0 = outid_list[0] # TODO 输出信息只有一个
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
    post_ctrresult(data_timelist1, userid1)
    post_ctrresult(data_timelist2, userid2)

def post_result_1(datalist1, end_index, idlist0):
    '''
    将当前时刻下的结果post到数据库中
    :param datalist1: 当前时刻下各区域光伏、负荷的准确率和最大误差, 优化前后的成本提升
    :param end_index: 当前时刻的index
    :return: None
    '''
    eqmnum0 = 'eqm000000'
    mytoday = date.today().strftime("%Y-%m-%d")
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    payload = {}
    for inds in range(len(datalist1)):
        key1 = eqmnum0 + ':ai:' + idlist0[inds]
        payload[key1] = {'v': str(datalist1[inds]),
                         't': mytoday + ' ' + timestamp_list96[end_index]}
    payload = json.dumps(payload)
    requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payload, headers=headers)
    requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)

def start_func(pre_load, pre_ne, pre_es, input_time_main, minute_div, SOC, my_end_inds, p_past_load,
               p_past_ne, p_past_es, c1, Pres, Eres, B, optTarget):
    '''
    获取策略的输入值，实际购售电量没用到
    :param pre_load: 预测负荷
    :param pre_ne: 预测光伏出力
    :param pre_es: 预测策略
    :param input_time_main: 当前时刻
    :param minute_div: 时间间隔
    :param SOC: 储能设备SOC
    :param my_end_inds: 当前时刻index
    :param p_past_load: 真实负荷
    :param p_past_ne: 真实光伏出力
    :param p_past_es: 真实策略
    :param c1: 购电价格
    :param Pres: 储能充放电功率
    :param Eres: 储能最大容量
    :param B: 调正系数
    :param optTarget: 策略目标：成本最优、碳排最优
    :return: 最优策略
    '''
    Pch_b = -1*np.clip(pre_es, -500, 0)
    Pdis_b = np.clip(pre_es, 0, 500)

    # -------------------------------已调整----------------------------------------
    p_past_load = np.array(p_past_load)  # 实际负荷出力数据--采集输入
    p_past_ne = np.array(p_past_ne)  # 实际新能源出力数据--采集输入

    # TODO 实际购售电量，没有用到
    e_past_sell, e_past_buy = get_dianliang(input_time_main, minute_div)
    e_past_sell = np.array(e_past_sell)
    e_past_buy = np.array(e_past_buy)
    # -------------------------------已调整----------------------------------------
    Pch_re_b = -1*np.clip(p_past_es, -500, 0)
    Pdis_re_b = np.clip(p_past_es, 0, 500)

    #TODO 目前是将实际数据和之前预测的数据结合，输入策略模型中计算控制策略，后续需要改成重新预测load和power的值，再输入进行预测
    loadB_Input = np.concatenate([p_past_load[:my_end_inds], pre_load[my_end_inds:]])
    powerB_Input = np.concatenate([p_past_ne[:my_end_inds], pre_ne[my_end_inds:]])

    optInputB = [loadB_Input, powerB_Input, SOC[0] / 100]
    [P_Be_b, P_out_b] = cal_new_stra(pre_load, pre_ne, Pch_b, Pdis_b, p_past_load, p_past_ne, list(map(lambda x: x/100, SOC)),
                                     Pch_re_b, Pdis_re_b, c1, Pres, Eres, pre_es, B,
                                     optTarget, optInputB)
    return [P_Be_b, P_out_b]

def env_sim(realload:list, realne:list, power_charge_cost:list, charge:list, discharge:list, Price_sold=0.3, count=4, qp=0.604):
    '''
    根据真实的负荷、出力、电价，根据不同的充放电策略，输出经济成本和碳排成本
    :param realload: 真实的负荷
    :param realne:  真实的光伏出力
    :param power_charge_cost: 电价购买
    :param charge: 储能充电策略
    :param discharge: 储能放电策略
    :return:
    '''
    buy_list = list(map(lambda x, y, m, n: 0 if (x + m) - (y + n) < 0 else (x + m) - (y + n),  realload, realne, charge, discharge))
    sell_list = list(map(lambda x, y, m, n: 0 if (x + m) - (y + n) < 0 else (x + m) - (y + n),  realne, realload, discharge, charge))
    z1 = sum(list(map(lambda x, y: x*y,  buy_list, power_charge_cost)))/count
    z2 = sum(sell_list) * Price_sold / count
    return z1 - z2, qp * np.sum(buy_list) / count

def myfunc():
    '''
    读取数据，在不同模式下计算最优策略，并post到数据库
    :return: None
    '''
    logger.info('本轮计算开始')
    sttime = time.time()
    input_time_main = datetime.datetime.now()
    count = 4
    minute_div = 60//count

    # 获取真实数据
    my_end_inds = input_time_main.hour*count + input_time_main.minute//minute_div
    # TODO 接口有问题， begin:'2023-06-23 00:00:00' end:'2023-06-23 00:01:00' 获取数据条数为0，秒数为0，获取数据计数少1
    # 获取真实数据，当前天0点到当前时刻的数据，时间间隔为15分钟
    [p_past_es1, p_past_load1, p_past_ne1, p_past_es2, p_past_load2, p_past_ne2, SOC1,
     SOC2] = get_today_real(input_time_main, minute_div)
    # pastdata = [p_past_es1, p_past_load1, p_past_ne1, p_past_es2, p_past_load2, p_past_ne2, soc1_past, soc2_past]
    # for i in range(0, len(pastdata)):
    #     print(i, len(pastdata[i]))
    modelparas = optimization_main.model_define()
    model_amount = modelparas[1]
    # print(model_amount)
    # TODO 模型参数输入
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
        # 获取预测数据
        predatas = get_today_pre(modelIndex, input_time_main)
        # 以下分别为1区预测负荷、光伏出力、风机出力、储能功率（正为放电、负为充电），均为list类型，长度96
        pre_load1 = predatas[0]
        pre_pv1 = predatas[1]
        pre_wind1 = predatas[2]
        pre_es1 = predatas[3]
        pre_ne1 = optimization_main.myplusfunc([pre_pv1, pre_wind1])
        # 以下分别为2区预测负荷、光伏出力、储能功率（正为放电、负为充电），均为list类型，长度96
        pre_load2 = predatas[4]
        pre_pv2 = predatas[5]
        pre_es2 = predatas[6]
        # 计算真实值和预测值的差值，并post
        def cal_correct_rate(real_list, pre_list):
            len_list = len(real_list)
            tmp_list = list(map(lambda x, y: np.abs(x-y), real_list, pre_list[:len_list]))
            list_2 = list(map(lambda m: 1 if m/np.average(pre_list) > 0.2 else 0, tmp_list))
            return sum(list_2)/len_list*100, max(tmp_list)
        load1_cor, load1_max_difference = cal_correct_rate(p_past_load1, pre_load1)
        load2_cor, load2_max_difference = cal_correct_rate(p_past_load2, pre_load2)
        ne1_cor, ne1_max_difference = cal_correct_rate(p_past_ne1, pre_pv1)
        ne2_cor, ne2_max_difference = cal_correct_rate(p_past_ne2, pre_pv2)
        post_result_1([load1_cor, load2_cor, load1_max_difference, load2_max_difference,
                       ne1_cor, ne2_cor, ne1_max_difference, ne2_max_difference],
                      my_end_inds, out_1_id_list[modelIndex])
        # 计算日前预测储能策略和两充两放储能策略的经济成本和碳排成本
        N1_cost1_pre, N1_cost2_pre = env_sim(p_past_load1, p_past_ne1, c1, -1*pre_es1.clip(-500, 0), pre_es1.clip(0, 500))
        N2_cost1_pre, N2_cost2_pre = env_sim(p_past_load2, p_past_ne2, c1, -1 * pre_es2.clip(-500, 0),
                                       pre_es2.clip(0, 500))
        sim_c, sim_stra = env_soc()
        N1_cost1_sim, N1_cost2_sim = env_sim(p_past_load1, p_past_ne1, c1, -1 * np.array(sim_stra).clip(-500, 0),
                                       np.array(sim_stra).clip(0, 500))
        N2_cost1_sim, N2_cost2_sim = env_sim(p_past_load2, p_past_ne2, c1, -1 * np.array(sim_stra).clip(-500, 0),
                                       np.array(sim_stra).clip(0, 500))
        post_result_1([N1_cost1_pre, N2_cost1_pre, N1_cost2_pre, N2_cost2_pre,
                       N1_cost1_sim, N2_cost1_sim, N1_cost2_sim, N2_cost2_sim],
                      my_end_inds, out_3_id_list[modelIndex])

        # 更新 根据不同区域计算更新储能控制策略
        if optObject == '1#储能':
            [P_Be_a, P_out_a] = start_func(pre_load1, pre_ne1, pre_es1, input_time_main, minute_div, SOC1, my_end_inds,
                       p_past_load1, p_past_ne1, p_past_es1, c1, Pres, Eres, B, optTarget)
            N1_cost1_opt, N1_cost2_opt = env_sim(p_past_load1, p_past_ne1, c1, -1 * np.array(P_Be_a).clip(-500, 0),
                                                 np.array(P_Be_a).clip(0, 500))
            post_result_1([N1_cost1_opt, N1_cost2_opt],
                          my_end_inds, out_2_id_list[modelIndex][::2])
            st = time.time()
            post_ctrresult(P_Be_a, outid_list[modelIndex][0])
            print('写数耗时', time.time() - st)
        if optObject == '2#储能':
            [P_Be_b, P_out_b] = start_func(pre_load2, pre_pv2, pre_es2, input_time_main, minute_div, SOC2, my_end_inds,
                       p_past_load2, p_past_ne2, p_past_es2, c1, Pres, Eres, B, optTarget)
            N2_cost1_opt, N2_cost2_opt = env_sim(p_past_load1, p_past_ne1, c1, -1 * np.array(P_Be_b).clip(-500, 0),
                                                 np.array(P_Be_b).clip(0, 500))
            post_result_1([N2_cost1_opt, N2_cost2_opt],
                          my_end_inds, out_2_id_list[modelIndex][1::2])
            st = time.time()
            post_ctrresult(P_Be_b, outid_list[modelIndex][1])
            print('写数耗时', time.time() - st)
        if optObject == '全选':
            [P_Be_a, P_out_a] = start_func(pre_load1, pre_ne1, pre_es1, input_time_main, minute_div, SOC1, my_end_inds,
                       p_past_load1, p_past_ne1, p_past_es1, c1, Pres, Eres, B, optTarget)
            [P_Be_b, P_out_b] = start_func(pre_load2, pre_pv2, pre_es2, input_time_main, minute_div, SOC2, my_end_inds,
                       p_past_load2, p_past_ne2, p_past_es2, c1, Pres, Eres, B, optTarget)
            N1_cost1_opt, N1_cost2_opt = env_sim(p_past_load1, p_past_ne1, c1, -1 * np.array(P_Be_a).clip(-500, 0),
                                                 np.array(P_Be_a).clip(0, 500))
            N2_cost1_opt, N2_cost2_opt = env_sim(p_past_load1, p_past_ne1, c1, -1 * np.array(P_Be_b).clip(-500, 0),
                                                 np.array(P_Be_b).clip(0, 500))
            post_result_1([N1_cost1_opt, N2_cost1_opt, N1_cost2_opt, N2_cost2_opt],
                          my_end_inds, out_2_id_list[modelIndex])
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
