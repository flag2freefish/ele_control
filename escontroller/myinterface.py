import time

import requests
import configparser
import json
import logging
from logging import handlers
from logging.handlers import TimedRotatingFileHandler
import sys
import os

run_mode = 'remot'
if run_mode == 'remot':
    url0 = 'http://123.60.30.122:21330'
    url_define = 'http://123.60.30.122:21306'
else:
    url0 = 'http://127.0.0.1:21530'
    url_define = 'http://127.0.0.1:21506'

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
# # 本地日志设置

# # # 设置日志文件路径
# # 获取系统环境变量
# delimeter = os.sep
# # env_path = os.getenv('XQNY_ROOT')
# # if not env_path:
# #     env_path = delimeter + 'home' + delimeter + 'iedp'
# # 获取程序名
# program_name = os.path.basename(sys.argv[0])
# # 生成路径
# # log_path = env_path + delimeter + 'calc' + delimeter + 'logs' + delimeter + program_name
# log_path = delimeter + program_name
# if not os.path.isdir(log_path):
#     os.makedirs(log_path)
# # 配置日志保存方式、文件名
# filename = 'D:/ProgramData/mylogs/mylog'
# formater = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
# timehandler = TimedRotatingFileHandler(filename,
#                                        when='D',  # 按日分割
#                                        interval=1,
#                                        backupCount=15,  # 保存15天
#                                        encoding='utf-8'
#                                        )
# timehandler.setFormatter(formater)
# timehandler.suffix = "%Y-%m-%d.log"  # 文件日期后缀
# logger = logging.getLogger('my_app')
# logger.setLevel(logging.INFO)
# logger.addHandler(timehandler)

# begin0 = "2022-08-27 00:00:00"
# end0 = "2022-08-27 23:59:59"
# dura0 = "1h"
# eqmnum0 = "eqm008001"
# dataidlist0 = ["QXZ000001", "QXZ000005"]


# begin:序列起始时间 end:序列终止时间  时间格式：%Y-%m-%d %H:%M:%S
# eqmnum:设备编号
# dataidlist: 待获取参数的点名列表，如['bms010001','bms010002']
def get_tsdb_core_ai(begin, end, dura, eqmnum, dataidlist):
    duration = '"duration":' + '"' + dura + '"'
    begin_time = '"begin_time":' + '"' + begin + '"'
    end_time = '"end_time":' + '"' + end + '"'
    point_list = '"point_list":{' + '"' + eqmnum + '":' + '['
    if len(dataidlist) == 0:
        raise SyntaxError
    elif len(dataidlist) == 1:
        point_list += '"' + dataidlist[-1] + '"]}'
    else:
        for inds in range(len(dataidlist) - 1):
            point_list += '"' + dataidlist[inds] + '",'
        point_list += '"' + dataidlist[-1] + '"]}'
    para = '{' + duration + ',' + begin_time + ',' + end_time + ',' + point_list + '}'
    querystring = "jsonStr=" + para
    url = url0 + "/" + tsdb_core_get_ai + "?" + querystring
    # print(url)
    # print(url)
    response = requests.get(url)
    response = response.json()
    # print(response)
    re = list()
    if response['data']:
        for j in range(0, len(dataidlist)):
            indstemp = -1
            for k in range(0, len(response['data'])):
                if response['data'][k]['point'][-9:] == dataidlist[j]:
                    indstemp = k
                    break

            if indstemp == -1:
                re.append([0])
            else:
                tempdata = list()
                for i in range(len(response['data'][indstemp]['list']) - 1):
                    tempdata.append(response['data'][indstemp]['list'][i]['v'])
                re.append(tempdata)
    return re


# begin:序列起始时间 end:序列终止时间  时间格式：%Y-%m-%d %H:%M:%S
# eqmnum:设备编号
# dataidlist: 待获取参数的点名列表，如['pcs010001','pcs010002']
def get_tsdb_core_di(begin, end, dura, eqmnum, dataidlist):
    duration = '"duration":' + '"' + dura + '"'
    begin_time = '"begin_time":' + '"' + begin + '"'
    end_time = '"end_time":' + '"' + end + '"'
    point_list = '"point_list":{' + '"' + eqmnum + '":' + '['
    if len(dataidlist) == 0:
        raise SyntaxError
    elif len(dataidlist) == 1:
        point_list += '"' + dataidlist[-1] + '"]}'
    else:
        for inds in range(len(dataidlist) - 1):
            point_list += '"' + dataidlist[inds] + '",'
        point_list += '"' + dataidlist[-1] + '"]}'
    para = '{' + duration + ',' + begin_time + ',' + end_time + ',' + point_list + '}'
    querystring = "jsonStr=" + para
    url = url0 + "/" + tsdb_core_get_di + "?" + querystring
    print(url)
    # print(url)
    response = requests.get(url)
    response = response.json()
    # print(response)
    re = list()
    if response['data']:
        for j in range(len(response['data'])):
            tempdata = list()
            for i in range(len(response['data'][j]['list']) - 1):
                tempdata.append(response['data'][j]['list'][i]['v'])
            re.append(tempdata)
    return re


# begin:序列起始时间 end:序列终止时间  时间格式：%Y-%m-%d %H:%M:%S
# eqmnum:设备编号
# dataidlist: 待获取参数的点名列表，如['opt010001','opt010002']
def get_tsdb_user_ai(begin, end, dura, eqmnum, dataidlist):
    duration = '"duration":' + '"' + dura + '"'
    begin_time = '"begin_time":' + '"' + begin + '"'
    end_time = '"end_time":' + '"' + end + '"'
    point_list = '"point_list":{' + '"' + eqmnum + '":' + '['
    if len(dataidlist) == 0:
        raise SyntaxError
    elif len(dataidlist) == 1:
        point_list += '"' + dataidlist[-1] + '"]}'
    else:
        for inds in range(len(dataidlist) - 1):
            point_list += '"' + dataidlist[inds] + '",'
        point_list += '"' + dataidlist[-1] + '"]}'
    para = '{' + duration + ',' + begin_time + ',' + end_time + ',' + point_list + '}'
    querystring = "jsonStr=" + para
    url = url0 + "/" + tsdb_user_get_ai + "?" + querystring
    # print(url)
    response = requests.get(url)
    response = response.json()
    # print(response)
    re = list()
    if response['data']:
        for j in range(0, len(dataidlist)):
            indstemp = -1
            for k in range(0, len(response['data'])):
                if response['data'][k]['point'][-9:] == dataidlist[j]:
                    indstemp = k
                    break

            if indstemp == -1:
                re.append([0])
            else:
                tempdata = list()
                for i in range(len(response['data'][indstemp]['list']) - 1):
                    tempdata.append(response['data'][indstemp]['list'][i]['v'])
                re.append(tempdata)
    return re


# begin:序列起始时间 end:序列终止时间  时间格式：%Y-%m-%d %H:%M:%S
# eqmnum:设备编号
# dataidlist: 待获取参数的点名列表，如['bms010001','bms010002']
def get_tsdb_user_di(begin, end, dura, eqmnum, dataidlist):
    duration = '"duration":' + '"' + dura + '"'
    begin_time = '"begin_time":' + '"' + begin + '"'
    end_time = '"end_time":' + '"' + end + '"'
    point_list = '"point_list":{' + '"' + eqmnum + '":' + '['
    if len(dataidlist) == 0:
        raise SyntaxError
    elif len(dataidlist) == 1:
        point_list += '"' + dataidlist[-1] + '"]}'
    else:
        for inds in range(len(dataidlist) - 1):
            point_list += '"' + dataidlist[inds] + '",'
        point_list += '"' + dataidlist[-1] + '"]}'
    para = '{' + duration + ',' + begin_time + ',' + end_time + ',' + point_list + '}'
    querystring = "jsonStr=" + para
    url = url0 + "/" + tsdb_core_get_di + "?" + querystring
    print(url)
    # print(url)
    response = requests.get(url)
    response = response.json()
    # print(response)
    re = list()
    if response['data']:
        for j in range(len(response['data'])):
            tempdata = list()
            for i in range(len(response['data'][j]['list']) - 1):
                tempdata.append(response['data'][j]['list'][i]['v'])
            re.append(tempdata)
    return re


# 待获取参数的全点名列表，如['eqm000001:ai:bms010001','eqm000001:di:bms010001']
def get_rtdb_core(useridlist):
    para = '['
    if len(useridlist) == 0:
        raise SyntaxError
    elif len(useridlist) == 1:
        para += '"' + useridlist[0] + '"]'
    else:
        for i in range(len(useridlist) - 1):
            para += '"' + useridlist[i] + '",'
        para += '"' + useridlist[-1] + '"]'
    querystring = "jsonStr=" + para
    url = url0 + "/" + rtdb_core_get_all + "?" + querystring
    # print(url)
    response = requests.get(url)
    response = response.json()
    re = list()
    if response['data']:
        for i in range(0, len(useridlist)):
            for j in range(len(response['data'])):
                if response['data'][j]['point'] == useridlist[i]:
                    re.append(response['data'][j]['v'])
                    break
    return re


# 待获取参数的全点名列表，如['eqm000001:ai:opt010001','eqm000001:di:opt010001']
def get_rtdb_user(useridlist):
    para = '['
    if len(useridlist) == 0:
        raise SyntaxError
    elif len(useridlist) == 1:
        para += '"' + useridlist[0] + '"]'
    else:
        for i in range(len(useridlist) - 1):
            para += '"' + useridlist[i] + '",'
        para += '"' + useridlist[-1] + '"]'
    querystring = "jsonStr=" + para
    url = url0 + "/" + rtdb_user_get_all + "?" + querystring
    print(url)
    response = requests.get(url)
    response = response.json()
    re = list()
    if response['data']:
        for i in range(0, len(useridlist)):
            for j in range(len(response['data'])):
                if response['data'][j]['point'] == useridlist[i]:
                    re.append(response['data'][j]['v'])
                    break
    return re


# datalist: 待写入的数据列表
# eqmlist: 与待写入数据对应的设备号
# idlist: 与待写入数据对应的点号
# timelist: 与待写入数据对应的时标，格式如'2022-08-31 18:00:00.000'
def post_ai(datalist, eqmlist, idlist, timelist):
    length = len(datalist)
    if len(eqmlist) != length or len(idlist) != length:
        raise SyntaxError
    else:
        headers = {
            'Content-Type': "application/json",
            'cache-control': "no-cache"
        }
        payload = {}
        for inds in range(length):
            # if idlist[inds][:3] == 'frm':
            #     key = eqmlist[inds] + ':calc_ai:' + idlist[inds]
            # else:
            #     key = eqmlist[inds] + ':ai:' + idlist[inds]
            key = eqmlist[inds] + ':ai:' + idlist[inds]
            payload[key] = {'v': str(datalist[inds]),
                            't': timelist[inds]}
        payload = json.dumps(payload)
        print('payload', payload)
        # posturl0 = 'http://123.60.30.122:21330'
        re1 = requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payload, headers=headers)
        re1 = re1.json()
        # print(url0 + '/' + tsdb_user_set_ai)
        re2 = requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)
        re2 = re2.json()
        print(url0 + '/' + tsdb_user_set_ai, url0 + '/' + rtdb_user_set_all)
        print('post_response', re1, re2)


def post_ai_new(datalist, eqmlist, idlist, timelist):
    length = len(datalist)
    if len(eqmlist) != length or len(idlist) != length:
        raise SyntaxError
    else:
        headers = {
            'Content-Type': "application/json",
            'cache-control': "no-cache"
        }
        payloadlist = list()
        for inds in range(length):
            payload = {'eqmid': eqmlist[inds] + ':ai:' + idlist[inds],
                       'v': str(datalist[inds]),
                       't': timelist[inds]}
            payloadlist.append(payload)

        payloadrt = {}
        for inds in range(length):
            # if idlist[inds][:3] == 'frm':
            #     key = eqmlist[inds] + ':calc_ai:' + idlist[inds]
            # else:
            #     key = eqmlist[inds] + ':ai:' + idlist[inds]
            key = eqmlist[inds] + ':ai:' + idlist[inds]
            payloadrt[key] = {'v': str(datalist[inds]),
                              't': timelist[inds]}

        payloadrt = json.dumps(payloadrt)
        payloadlist = json.dumps(payloadlist)
        print('payload', payloadlist)
        print('payload', payloadrt)
        re1 = requests.request("POST", url0 + '/' + tsdb_user_set_ai, data=payloadlist, headers=headers)
        re1 = re1.json()
        # print(url0 + '/' + tsdb_user_set_ai)
        re2 = requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payloadrt, headers=headers)
        re2 = re2.json()
        # print(url0 + '/' + tsdb_user_set_ai, url0 + '/' + rtdb_user_set_all)
        print('post_response', re1, re2)


# datalist: 待写入的数据列表
# eqmlist: 与待写入数据对应的设备号
# idlist: 与待写入数据对应的点号
# timelist: 与待写入数据对应的时标，格式如'2022-08-31 18:00:00.000'
def post_di(datalist, eqmlist, idlist, timelist):
    length = len(datalist)
    if len(eqmlist) == length and len(idlist) == length:
        raise SyntaxError
    else:
        headers = {
            'Content-Type': "application/json",
            'cache-control': "no-cache"
        }
        payload = {}
        for inds in range(length):
            key = eqmlist[inds] + ':di:' + idlist[inds]
            payload[key] = {'v': str(datalist[inds]),
                            't': timelist[inds]}
        payload = json.dumps(payload)
        requests.request("POST", url0 + '/' + tsdb_user_set_di, data=payload, headers=headers)
        requests.request("POST", url0 + '/' + rtdb_user_set_all, data=payload, headers=headers)

# myre = get_tsdb_core_ai(begin0, end0, dura0, eqmnum0, dataidlist0)
# print(myre)
# logger.info('测试')
# print(get_rtdb_core(["eqm007001:ai:BMS000008", "eqm007002:ai:BMS000008"]))
