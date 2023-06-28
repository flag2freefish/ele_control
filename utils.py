import numpy as np
import pandas as pd

def plot(Pes_b, Pch_re_b, Pdis_re_b, P_Be_b):
    from matplotlib import pyplot as plt
    plt.plot(Pes_b, label='first_stra')
    plt.plot(Pch_re_b, label='real_charge')
    plt.plot(Pdis_re_b, label='real_discharge')
    plt.plot(P_Be_b, label='second_stra')
    plt.legend()

def env_soc(count=4):
    a = [-250]*12*2 + [0]*12*8 + [250]*12*2 + [0]*12 + [-250] *12*2 + [0]*12*2 + [250] *12*2 +[0]*12*6

    b = list(map(lambda x: (x/24)*100+1, range(24))) + [99]*12*3 \
        + list(map(lambda x: (x/24)*100-1, range(24,0, -1))) + [1] * 12 \
        + list(map(lambda x: (x/24)*100+1, range(24))) + [99] * 12*2 \
        + list(map(lambda x: (x/24)*100-1, range(24,0, -1))) + [1] * 12*6
    tmp_div = int(12/count)
    return a[::tmp_div].copy(), b[::tmp_div].copy()

def get_sim_soc_stra(curent_time, count=4):
    cu_time = pd.to_datetime(curent_time)
    min_count = cu_time.minute//(60/count)

    cu_index = int(cu_time.hour * count + min_count)
    a, b = env_soc(count)
    return a[:cu_index], b[:cu_index]

def env_sim(realload:list, realne:list, power_charge_cost:list, charge:list, discharge:list, Price_sold=0.3, count=3):
    '''
    在过去24小时的时间内，根据真实的负荷、出力、电价，根据不同的充放电策略，输出过去24小时的经济成本和碳排成本
    :param realload:
    :param realne:
    :param power_charge_cost:
    :param charge:
    :param discharge:
    :return:
    '''
    buy_list = list(map(lambda x, y, m, n: 0 if (x + m) - (y + n) < 0 else (x + m) - (y + n),  realload, realne, charge, discharge))
    sell_list = list(map(lambda x, y, m, n: 0 if (x + m) - (y + n) < 0 else (x + m) - (y + n),  realne, realload, discharge, charge))
    z1 = sum(list(map(lambda x, y: x*y,  buy_list, power_charge_cost)))/count
    z2 = sum(sell_list) * Price_sold
    return z1 - z2, 0.604 * np.sum(buy_list) / count

def realstra(Pload, realload, Pne, realne, predstra, soc, power_charge_cost):
    '''
    输入预测策略，预测负荷、出力，和真实的负荷、出力，计算调整后的策略

    :param Pload:
    :param realload:
    :param Pne:
    :param realne:
    :param predstra:
    :return:
    '''
    if len(Pload) == 24:
        div_num = 1
    else:
        div_num = len(Pload)/24
    a_load = list(map(lambda x, y: np.abs(x-y), realload, Pload))
    mean_Pload = np.mean(Pload)
    a_load = a_load/mean_Pload
    count1load = list(map(lambda x: 1 if x > 0.2 else 0, a_load))
    count2load = list(map(lambda x: 1 if x > 0.5 else 0, a_load))
    a_pne = list(map(lambda x, y: np.abs(x - y), realne, Pne))
    mean_Pne = np.mean(Pne)
    a_ne = a_pne / mean_Pne
    count1ne = list(map(lambda x: 1 if x > 0.2 else 0, a_ne))
    count2ne = list(map(lambda x: 1 if x > 0.5 else 0, a_ne))
    count1 = list(map(lambda x, y: 1 if x or y else 0, count1load, count1ne))
    count2 = list(map(lambda x, y: 1 if x or y else 0, count2load, count2ne))
    if sum(count1)/div_num < 20 and sum(count2)/div_num < 32:
        # 实时的新能源供电误差和负荷误差之间的差值不超过0.2，输出原来的策略，（未来24个小时，96或288个点位的策略）
        # 差值超过0.2，使用策略修改器修改
        # TODO 这个差值直接用前面的差值计算，量纲不同
        if np.abs(a_load[-1] - a_ne[-1]) > 0.2:
            if realne[-1] > realload[-1]:
                if soc > 0.9:
                    predstra[0] = 0
                else:
                    predstra[0] = realne[-1] - realload[-1]
            else:
                if power_charge_cost[-1] > min(power_charge_cost):
                    if soc > 0.2:
                        predstra[0] = realne[-1] - realload[-1]
                    else:
                        predstra[0] = 0
                else:
                    if soc > 0.9:
                        predstra[0] = 0
                    else:
                        predstra[0] = predstra[0] * (1 + realload[-1] - realne[-1])

        return predstra
    else:
        return repredict()