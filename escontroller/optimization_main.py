# 仅用于引用
import requests
import json
import numpy as np
import pulp

# from loadApark_pre import mypre_loadApark
# from loadApile_pre import mypre_loadApile
# from loadAld_pre import mypre_loadAld
# from loadAzm_pre import mypre_loadAzm
# from loadBpark_pre import mypre_loadBpark
# from loadBpile_pre import mypre_loadBpile
# from loadBld_pre import mypre_loadBld
# from loadBzm_pre import mypre_loadBzm

serviceNum = '010011'
# print('serviceNum', serviceNum)

run_mode = 'remot'
if run_mode == 'remot':
    url0 = 'http://123.60.30.122:21330'
    url_define = 'http://123.60.30.122:21306'
else:
    url0 = 'http://127.0.0.1:21530'
    url_define = 'http://127.0.0.1:8006'
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

url_get_define = url_define + '/' + 'Calclation/findInfoCustomList'


def model_define():
    payload = {'serviceNum': serviceNum}
    payload = json.dumps(payload)
    # print('payload', payload)
    headers = {
        'Content-Type': 'application/json'
    }
    # print('url_get_define', url_get_define)
    # print('payload', payload)
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


# 获取取值范围
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
# print(timestamp_list)
# print(len(timestamp_list))


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
    f.solve()
    # for i in f.variables():
    #     print(i.name,"=",i.varValue)
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
    c += pulp.lpSum(Pbuy) * qn / 4
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
    f += np.sum(Pbuy[0:int(T) - 1]) == c * 4 * qn1
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
    output_stra = Pes
    optout_profit = profit
    output_stra24 = list()
    for i in range(24):
        output_stra24.append(output_stra[4*i])
    output_stra288 = myinterpolation(output_stra24, '288')
    if amount == 96:
        return output_stra, optout_profit
    elif amount == 24:
        return output_stra24, optout_profit
    else:
        return output_stra288, optout_profit
    # output_stra = [1]*288
    # optout_profit = [99]*3
    # if amount == 96:
    #     output_stra = [1] * 96
    # elif amount == 24:
    #     output_stra = [1] * 24
    # print('optout_profit', optout_profit)


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
        output_stra24.append(output_stra[4*i])
    output_stra288 = myinterpolation(output_stra24, '288')
    if amount == 96:
        return output_stra, optout_profit
    elif amount == 24:
        return output_stra24, optout_profit
    else:
        return output_stra288, optout_profit
    # output_stra = [1] * 288
    # optout_profit = [99] * 3
    # if amount == 96:
    #     output_stra = [1] * 96
    # elif amount == 24:
    #     output_stra = [1] * 24
    # return output_stra, optout_profit


