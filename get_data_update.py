# -*- coding: utf-8 -*-
# @Time : 2023/4/18 14:27
# @Author : Ziyao Wang
# @FileName: get_data_update.py
# @Describe:
import configparser
import requests

config = configparser.ConfigParser()
ini_path = 'opt_config.ini'
config.read(ini_path, encoding='utf-8')

# run_mode = 'remot'
run_mode = config.get('run_mode', 'run_mode_ind')
if run_mode == 'remot':
    url0 = 'http://123.60.30.122:21330/'
    # url0 = config.get('url_remot', 'url0')
else:
    url0 = config.get('url_local', 'url0')
get_core_tsdb = 'tsdb_core_get_ai_by_eqm_and_id'
get_core_rtdb = 'rtdb_core_get_points_by_eqmid'
get_user_tsdb = 'tsdb_user_get_ai_by_eqmid_and_id'
set_user_tsdb_ai = 'tsdb_user_set_ai_by_eqmid'
set_user_rtdb = 'rtdb_user_set_points_by_eqmid'

A_park_plus = config.get('keys_load', 'A_park_plus')
A_park_minus = config.get('keys_load', 'A_park_minus')
A_park_plus_list = A_park_plus.split(',')
A_park_minus_list = A_park_minus.split(',')
A_pile = config.get('keys_load', 'A_pile')
A_ludeng = config.get('keys_load', 'A_ludeng')
A_zhiliuzm = config.get('keys_load', 'A_zhiliuzm')

B_park_plus = config.get('keys_load', 'B_park_plus')
B_park_minus = config.get('keys_load', 'B_park_minus')
B_park_plus_list = B_park_plus.split(',')
B_park_minus_list = B_park_minus.split(',')
B_park = config.get('keys_load', 'B_park')
B_pile = config.get('keys_load', 'B_pile')
B_ludeng = config.get('keys_load', 'B_ludeng')
B_zhiliuzm = config.get('keys_load', 'B_zhiliuzm')


def get_power_pvA(begin, end):
    querystring = 'jsonStr={"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' + \
                  '"point_list":{"eqm003001":["PCS100040"],"eqm003002":["PCS100040"],"eqm003003":["PCS100040"],' \
                  '"eqm003004":["PCS100040"],"eqm003005":["PCS100040"],"eqm003006":["PCS100040"],' \
                  '"eqm002024":["DCM000003"],"eqm002025":["DCM000003"],"eqm002026":["DCM000003"]}}'# 单独取点查看数据
    url = url0 + get_core_tsdb + "?" + querystring

    print(url)
    response = requests.get(url)
    response = response.json()
    pvA_p = list()
    if response['data']:
        for j in range(len(response['data'][0]['list'])):
            temporary = 0
            for i in range(len(response['data'])):
                temporary += response['data'][i]['list'][j]['v']
            if temporary < 0:
                temporary = 0
            pvA_p.append(temporary)

    return pvA_p

def get_power_pvA_1(begin, end):
    querystring = 'jsonStr={"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' + \
                  '"point_list":{"eqm003001":["PCS100040"],"eqm003002":["PCS100040"],"eqm003003":["PCS100040"],' \
                  '"eqm003004":["PCS100040"],"eqm003005":["PCS100040"],"eqm003006":["PCS100040"],' \
                  '"eqm002024":["DCM000003"],"eqm002025":["DCM000003"],"eqm002026":["DCM000003"]}}'# 单独取点查看数据
    url = url0 + get_core_tsdb + "?" + querystring

    print(url)
    response = requests.get(url)
    response = response.json()
    all_data = {}
    if response['data']:
        for i in range(len(response['data'])):
            tmp_list = []
            for j in range(len(response['data'][i]['list'])):
                tmp_list.append(response['data'][i]['list'][j]['v'])
            all_data[response['data'][i]['point']] = tmp_list
    return all_data

def get_power_pvB(begin, end):
    querystring = 'jsonStr={"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' + \
                  '"point_list":{"eqm002038":["DCM000003"],' \
                  '"eqm003007":["PCS100040"],"eqm003008":["PCS100040"],"eqm003009":["PCS100040"],' \
                  '"eqm003010":["PCS100040"],"eqm003011":["PCS100040"],"eqm003012":["PCS100040"],' \
                  '"eqm008001":["PCS110004"],"eqm008003":["PCS120004"],' \
                  '"eqm008002":["PCS110004"],"eqm008004":["PCS110004"],' \
                  '"eqm002039":["DCM000003"],"eqm002037":["DCM000003"]}}'
    url = url0 + get_core_tsdb + "?" + querystring

    print(url)
    response = requests.get(url)
    response = response.json()
    pvB_p = list()
    if response['data']:
        for j in range(len(response['data'][0]['list'])):
            temporary = 0
            for i in range(len(response['data'])):
                temporary += response['data'][i]['list'][j]['v']
            if temporary < 0:
                temporary = 0
            pvB_p.append(temporary)
    return pvB_p


def get_userdb(begin, end, dura, eqmid, userid):
    # keylist1 = A_park_plus_list
    # keylist2 = A_park_minus_list
    # minus_num = len(keylist2)
    # keylist = keylist1 + keylist2
    para = '{"duration":"' + dura + '","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' \
           + '"point_list":' + '{'
    para += '"' + eqmid + '":' + '[' + '"' + userid + '"]}}'
    querystring = "jsonStr=" + para
    url = url0 + get_user_tsdb + "?" + querystring
    print(url)
    response = requests.get(url)
    response = response.json()
    redata = list()
    if response['data']:
        for j in range(0, len(response['data'][0]['list'])):
            temporary = response['data'][0]['list'][j]['v']
            # if temporary < 0:
            #     temporary = 0
            redata.append(temporary)
    # print(load_A_park)
    return redata


def get_load_A_park(begin, end):
    keylist1 = A_park_plus_list
    keylist2 = A_park_minus_list
    minus_num = len(keylist2)
    keylist = keylist1 + keylist2
    para = '{"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' \
           + '"point_list":' + '{'
    for i in range(len(keylist) - 1):
        point = '"' + keylist[i][0:9] + '":' + '[' + '"' + keylist[i][-9:] + '"],'
        para += point
    para += '"' + keylist[-1][0:9] + '":' + '[' + '"' + keylist[-1][-9:] + '"]}}'
    querystring = "jsonStr=" + para
    querystring = 'jsonStr={"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' + \
                  '"point_list":{"eqm003021":["ACM100013"],' \
                  '"eqm003018":["WTG000004"],"eqm003019":["WTG000004"],"eqm003020":["WTG000004"],' \
                  '"eqm003001":["PCS100040"],"eqm003002":["PCS100040"],"eqm003003":["PCS100040"],' \
                  '"eqm003004":["PCS100040"],"eqm003005":["PCS100040"],"eqm003006":["PCS100040"],' \
                  '"eqm003023":["ACM100013"],"eqm002043":["SWA000004"]}}'
    url = url0 + get_core_tsdb + "?" + querystring
    print(url)
    response = requests.get(url)
    response = response.json()
    load_A_park = list()
    if response['data']:
        # for i in range(0, len(response['data'])):
        #     print(len(response['data'][i]['list']))
        for j in range(0, len(response['data'][0]['list'])):
            temporary = 0
            for i in range(0, len(response['data'])):
                if response['data'][i]['point'] != "eqm002043:ai:SWA000004" and response['data'][i]['point'] != "eqm003023:ai:ACM100013":
                    temporary += response['data'][i]['list'][j]['v']
                else:
                    temporary -= response['data'][i - len(response['data'])]['list'][j]['v']
            # if temporary < 0:
            #     temporary = 0
            load_A_park.append(temporary)
    # print(load_A_park)
    return load_A_park


def get_load_A_others(begin, end):
    keylist_others = [A_pile, A_ludeng, A_zhiliuzm]
    if keylist_others[1] == 'none':
        keylist_others[1] = A_pile
    para = '{"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' \
           + '"point_list":' + '{'
    for i in range(0, len(keylist_others) - 1):
        point = '"' + keylist_others[i][0:9] + '":' + '[' + '"' + keylist_others[i][-9:] + '"],'
        para += point
    para += '"' + keylist_others[-1][0:9] + '":' + '[' + '"' + keylist_others[-1][-9:] + '"]}}'
    querystring = "jsonStr=" + para
    url = url0 + get_core_tsdb + "?" + querystring
    print(url)
    response = requests.get(url)
    response = response.json()
    load_A_pile = list()
    load_A_ludeng = list()
    load_A_zhiliuzm = list()
    if response['data']:
        for i in range(0, len(response['data'])):
            if response['data'][i]['point'] == A_pile:
                for j in range(len(response['data'][i]['list'])):
                    load_A_pile.append(response['data'][i]['list'][j]['v'])
            if response['data'][i]['point'] == A_ludeng:
                for j in range(len(response['data'][i]['list'])):
                    load_A_ludeng.append(response['data'][i]['list'][j]['v'])
            if response['data'][i]['point'] == A_zhiliuzm:
                for j in range(len(response['data'][i]['list'])):
                    load_A_zhiliuzm.append(response['data'][i]['list'][j]['v'])

    # print('load_A_pile', len(load_A_pile))
    # print('load_A_ludeng', len(load_A_ludeng))
    # print('load_A_zhiliuzm', len(load_A_zhiliuzm))
    return load_A_pile, load_A_ludeng, load_A_zhiliuzm


def get_load_B_park(begin, end):
    keylist1 = B_park_plus_list
    keylist2 = B_park_minus_list
    minus_num = len(keylist2)
    keylist = keylist1 + keylist2
    para = '{"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' \
           + '"point_list":' + '{'
    for i in range(len(keylist) - 1):
        point = '"' + keylist[i][0:9] + '":' + '[' + '"' + keylist[i][-9:] + '"],'
        para += point
    para += '"' + keylist[-1][0:9] + '":' + '[' + '"' + keylist[-1][-9:] + '"]}}'
    querystring = "jsonStr=" + para
    querystring = 'jsonStr={"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' + \
                  '"point_list":{"eqm003022":["ACM100013"],' \
                  '"eqm003007":["PCS100040"],"eqm003008":["PCS100040"],"eqm003009":["PCS100040"],' \
                  '"eqm003010":["PCS100040"],"eqm003011":["PCS100040"],"eqm003012":["PCS100040"],' \
                  '"eqm008001":["PCS110004"],"eqm008003":["PCS120004"],' \
                  '"eqm008002":["PCS110004"],"eqm008004":["PCS110004"],' \
                  '"eqm002044":["SWA000004"]}}'
    url = url0 + get_core_tsdb + "?" + querystring
    print(url)
    response = requests.get(url)
    response = response.json()
    load_B_park = list()
    if response['data']:
        for j in range(len(response['data'][0]['list'])):
            temporary = 0
            for i in range(len(response['data']) - minus_num):
                if response['data'][i]['point'] != "eqm002044:ai:SWA000004":
                    temporary += response['data'][i]['list'][j]['v']
                else:
                    temporary -= response['data'][i]['list'][j]['v']
            load_B_park.append(temporary)
    return load_B_park


def get_load_B_others(begin, end):
    keylist_others = [B_pile, B_ludeng, B_zhiliuzm]
    if keylist_others[1] == 'none':
        keylist_others[1] = B_pile
    para = '{"duration":"1h","end_time":' + '"' + end + '",' + '"begin_time":' + '"' + begin + '",' \
           + '"point_list":' + '{'
    for i in range(len(keylist_others) - 1):
        point = '"' + keylist_others[i][0:9] + '":' + '[' + '"' + keylist_others[i][-9:] + '"],'
        para += point
    para += '"' + keylist_others[-1][0:9] + '":' + '[' + '"' + keylist_others[-1][-9:] + '"]}}'
    querystring = "jsonStr=" + para
    url = url0 + get_core_tsdb + "?" + querystring
    response = requests.get(url)
    response = response.json()
    load_B_pile = list()
    load_B_ludeng = list()
    load_B_zhiliuzm = list()
    if response['data']:
        for i in range(0, len(response['data'])):
            if response['data'][i]['point'] == B_pile:
                for j in range(len(response['data'][i]['list'])):
                    load_B_pile.append(response['data'][i]['list'][j]['v'])
            if response['data'][i]['point'] == B_ludeng:
                for j in range(len(response['data'][i]['list'])):
                    load_B_ludeng.append(response['data'][i]['list'][j]['v'])
            if response['data'][i]['point'] == B_zhiliuzm:
                for j in range(len(response['data'][i]['list'])):
                    load_B_zhiliuzm.append(response['data'][i]['list'][j]['v'])
    return load_B_pile, load_B_ludeng, load_B_zhiliuzm
