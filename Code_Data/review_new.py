import numpy as np
import sys
import uuid
import time
import datetime
from odps.tunnel import TableTunnel
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                        数据处理部分                                                        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 1、提取数据库原始数据
nownow = datetime.datetime.now()
print(nownow)
print("start")

nowdatetime = str(args['datetimes'])
print (nowdatetime)
#nowdatetime = "20220929093000"

print("1 ========== " + nowdatetime)
nowdatetime = datetime.datetime.strptime(nowdatetime, "%Y%m%d%H%M%S")
nowdatetime = nowdatetime - datetime.timedelta(minutes=59)
nowdatetime = str(nowdatetime.strftime("%Y%m%d%H%M%S"))
print("2 ========== " + nowdatetime)
nowdate = nowdatetime[0:8]
nowtime = nowdatetime[8:14]
node_db_list = []
node_db_num = 0
asksql = 'select node_id, node_name, node_vl, node_type, node_cei, data_date, time, node_pg, b.node_p, area_num from hd_city_dev.self_ads_node_info a left join hd_city_pro.self_ads_node_data b on a.node_num = b.node_num  where  data_date = ''%s'' and b.time = ''%s'' and a.node_vl >= ''110'''%(nowdate,nowtime)
node_db = o.execute_sql(asksql,hints={'odps.sql.allow.fullscan':'true'})
#print(node_db)
with node_db.open_reader() as reader:
    for record in reader:
        node_db_list.insert(node_db_num,record)
        node_db_num = node_db_num+1
#print (node_db_list)
# line_db_list = 0:line_id 1:data_date 2:time 3:line_P 4:start_nid 5:end_nid 6:line_name
line_db_list = []
line_db_num = 0
asksql = 'select a.line_id,b.data_date,b.time,b.line_p,a.start_nid,a.end_nid,a.line_name from hd_city_dev.self_ads_line_info a LEFT JOIN hd_city_pro.self_ads_line_data b on a.LINE_NUM= b.line_num where   b.data_date = ''%s'' and b.time = ''%s'' and line_vl >= ''110'''%(nowdate,nowtime)
line_db = o.execute_sql(asksql,hints={'odps.sql.allow.fullscan':'true'})
with line_db.open_reader() as reader:
     for record in reader:
         line_db_list.insert(line_db_num,record)
         line_db_num = line_db_num+1
#print (line_db_list)
# area_db_list = 0:area_num 1:area_type 2:parent_id 3:area_level
area_db_list = []
area_db_num = 0
asksql = 'SELECT area_num, area_type, parent_id, area_level, area_id, area_name FROM hd_city_dev.self_ads_area_info WHERE area_level > ''0'';'
area_db = o.execute_sql(asksql,hints={'odps.sql.allow.fullscan':'true'})
with area_db.open_reader() as reader:
     for record in reader:
         area_db_list.insert(area_db_num,record)
         area_db_num = area_db_num+1
#print (area_db_list)
nownow = datetime.datetime.now()
print(nownow)
print("get data success")

# 2、数据库数据处理解析
# node_data_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id
#node_data_list = []
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
#line_data_list = []
# area_data_list = 0:area_id 1:area_num 2:area_type 3:parent_id 4:area_level 5:self_id 6:area_name
#area_data_list = []

# node_data_list数据生成
# number = 1
# for index in node_db_list:
#     temp = [index[0], number, index[3], index[7], index[4], index[9], index[8], 0]
#     node_data_list.append(temp)
#     # node_data_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag
#     number = number + 1
# 优化代码-node_data_list
b = range(1, len(node_db_list) + 1)
node_data_list = list(map(lambda x, y: [x[0], y, x[3], x[7], x[4], x[9], x[8], 0], node_db_list, b))


# line_data_list数据生成
# node_cal_list为剔除空节点后的节点数据列表
# node_cal_list = []
# number = 1
# for node_data_item in node_data_list:
#     for index in line_db_list:
#         if index[4] == node_data_item[0]:
#             if node_data_item not in node_cal_list:
#                 node_cal_list.append(node_data_item)
#         if index[5] == node_data_item[0]:
#             if node_data_item not in node_cal_list:
#                 node_cal_list.append(node_data_item)
#
# # 节点重新编号
# number = 1
# for index in node_cal_list:
#     index[1] = number
#     number = number + 1
#     # node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag

# 优化代码-node_cal_list
target_line_1 = list(map(lambda x: x[4], line_db_list))
target_line_2 = list(map(lambda x: x[5], line_db_list))
node_cal_list = []
for node_data_item in node_data_list:
    if node_data_item[0] in target_line_1 or node_data_item[0] in target_line_2:
        node_cal_list.append(node_data_item)
# 节点重排号
def replace1(x, y):
    x[1] = y
    return x
node_cal_list = list(map(lambda x, y: replace1(x, y), node_cal_list, b))

# 线路重新编号
# line_data_list = []
# number = 1
# print (line_db_list)
# print (node_cal_list)
# for index in line_db_list:
#     for node_data_item in node_cal_list:
#         if index[4] == node_data_item[0]:
#             start_num = node_data_item[1]
#         if index[5] == node_data_item[0]:
#             end_num = node_data_item[1]
#         #print (index[4])
#         #print (node_data_item[0])
#     temp = [index[0], number, start_num, end_num, index[3]]
#     line_data_list.append(temp)
#     # line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
#     number = number + 1

# 线路重新编号
node_data_item_0 = list(map(lambda x: x[0], node_cal_list))
node_data_item_1 = list(map(lambda x: x[1], node_cal_list))
b = list(range(1, len(line_db_list) + 1))
def x_y(x, y):
    if x[4] in node_data_item_0:
        return [x[0], y, node_data_item_1[node_data_item_0.index(x[4])], node_data_item_1[node_data_item_0.index(x[5])], x[3]]
line_data_list = list(map(lambda x, y:x_y(x, y), line_db_list, b))

# area_data_list = 0:area_id 1:area_num 2:area_type 3:parent_id 4:area_level 5:self_id 6:area_name
# node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id
# area_data_list = []
# number = 1
# for index in area_db_list:
#     area_data_list.append([index[0], number, index[1], index[2], int(index[3]), index[4], index[5]])
#     number = number + 1

# 优化代码-area_data_list
area_data_list = list(map(lambda x, y: [x[0], y, x[1], x[2], int(x[3]), x[4], x[5]], area_db_list, b))

# for node_item in node_cal_list:
#     for area_item in area_data_list:
#         if node_item[5] == area_item[0]:
#             node_item[5] = area_item[1]
# 优化代码-node_cal_list
area_data_item_0 = list(map(lambda x: x[0], area_data_list))
area_data_item_1 = list(map(lambda x: x[1], area_data_list))
def replace2(x):
    x[5] = area_data_item_1[area_data_item_0.index(x[5])]
    return x
node_cal_list = list(map(lambda x: replace2(x), node_cal_list))

nownow = datetime.datetime.now()
print(nownow)
print("data treatment")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       矩阵生成部分     
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
# node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei  5:area_num                                                                                                     
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
M = len(node_cal_list) # 节点数量
N = len(line_data_list) # 线路数量

# 矩阵初始化

P_G = np.zeros((M, M), dtype = np.float)
P_Z = np.zeros((2*M, M), dtype = np.float)
E_G = np.zeros((M, 1), dtype = np.float)
P_L = np.zeros((M, M), dtype = np.float)
P_B = np.ones((M, M), dtype = np.float)*0.00001

# 矩阵赋值

P_import = 0
for node_data_item in node_cal_list:
    tempdata = 0
    if int(node_data_item[2]) == 3:# if it is a electricity substation
        node_data_item[3] = 0
        node_data_item[4] = 0
        for line_data_item in line_data_list:
            if node_data_item[1] == line_data_item[2]:
                P_L[node_data_item[1]-1, node_data_item[1]-1] = P_L[node_data_item[1]-1, node_data_item[1]-1] - float(line_data_item[4])# 如果是某条线路的起始节点，为流出
            elif node_data_item[1] == line_data_item[3]:
                P_L[node_data_item[1]-1, node_data_item[1]-1] = P_L[node_data_item[1]-1, node_data_item[1]-1] + float(line_data_item[4])# 如果是某条线路的终止节点，为流入
        if P_L[node_data_item[1]-1, node_data_item[1]-1] < 0:#supply by import electricity
            node_data_item[4] = 0.5839
            node_data_item[3] = abs(P_L[node_data_item[1]-1, node_data_item[1]-1]) + abs(float(node_data_item[6]))
            P_L[node_data_item[1]-1, node_data_item[1]-1] = abs(float(node_data_item[6]))
            P_import = P_import + node_data_item[3]
            node_data_item[7] = 1
    else: # if it is not a electricity substation
        for line_data_item in line_data_list:
            if node_data_item[1] == line_data_item[2]:
                P_L[node_data_item[1]-1, node_data_item[1]-1] = P_L[node_data_item[1]-1, node_data_item[1]-1] - float(line_data_item[4])# 如果是某条线路的起始节点，为流出
            elif node_data_item[1] == line_data_item[3]:
                P_L[node_data_item[1]-1, node_data_item[1]-1] = P_L[node_data_item[1]-1, node_data_item[1]-1] + float(line_data_item[4])# 如果是某条线路的终止节点，为流入
        if P_L[node_data_item[1]-1, node_data_item[1]-1] < 0: #support the network
            node_data_item[3] = abs(float(node_data_item[6])) - P_L[node_data_item[1]-1, node_data_item[1]-1]
            P_L[node_data_item[1]-1, node_data_item[1]-1] = abs(float(node_data_item[6]))
        else:# supply by the network
            node_data_item[3] = abs(float(node_data_item[6])) - P_L[node_data_item[1]-1, node_data_item[1]-1]
            if node_data_item[3] < 0:
                node_data_item[3] = 0
            else:
                P_L[node_data_item[1]-1, node_data_item[1]-1] = abs(float(node_data_item[6]))

for line_data_item in line_data_list:
    n_p = float(line_data_item[4])
    if n_p >= 0:
        P_B[line_data_item[2]-1, line_data_item[3]-1] = n_p + P_B[line_data_item[2]-1, line_data_item[3]-1]
    else:
        P_B[line_data_item[3]-1, line_data_item[2]-1] = P_B[line_data_item[3]-1, line_data_item[2]-1] - n_p

for node_data_item in node_cal_list:
    if node_data_item[3] >= 0:
        P_G[node_data_item[1]-1, node_data_item[1]-1] = node_data_item[3]
    else:
        node_data_item[3] = 0

P_Z = np.vstack((P_B, P_G))

for node_data_item in node_cal_list:
    E_G[node_data_item[1]-1, 0] = node_data_item[4]

P_N = np.diag(np.dot(np.ones((1, 2*M), dtype=np.float), P_Z)[0,:])

nownow = datetime.datetime.now()
print(nownow)
print("matrix production")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       结果运算部分 
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
# node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_num                                                                                                          
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# NODE

E_N = np.dot(np.dot(np.linalg.inv(P_N - np.transpose(P_B)), np.transpose(P_G)), E_G)
C_U = np.dot(np.transpose(E_N), P_L)
C_G = np.dot(np.transpose(E_G), P_G)
node_result_list = []
number = 0
for index in node_cal_list:
    if(int(index[2]) == 3):
        node_result_list.append([index[0], index[1], E_N[number].tolist()[0], C_U[0, number].tolist(), 0, P_L[number, number], index[5], 0, index[7]])
    else:
        node_result_list.append([index[0], index[1], index[4], C_U[0, number].tolist(), C_G[0, number].tolist(), P_L[number, number], index[5], P_G[number, number], index[7]])
    number = number + 1
# node_result_list = 0:node_id 1:node_num 2:CEI 3:CEU 4:CEG 5:P_L 6:area_num

# LINE

line_result_list = []
number = 0
for line_data_item in line_data_list:
    for node_result_item in node_result_list:
        if line_data_item[2] == node_result_item[1]:
            temp1 = float(node_result_item[2]) * float(line_data_item[4]) / 6
            temp2 = float(node_result_item[2]) * float(line_data_item[4])
            line_result_list.append([line_data_item[0], line_data_item[1], node_result_item[2], temp1, temp2])
            number = number + 1


# line_result_list = 0:line_id 1:line_num 2:Line_CEI 3:Line_CEF 4:Line_CEFR
all_CEU = 0
all_CEG = 0
all_PL = 0
all_PG = 0
all_CEI = 0
all_CFI = 0
print(node_result_list)
for item in node_result_list:
    all_CEU = item[3] + all_CEU
    all_CEG = item[4] + all_CEG
    all_PL = item[5] + all_PL
    all_PG = item[7] + all_PG
if all_PL > 0:
    all_CEI = all_CEU / all_PL
else:
    print('error')
print(all_CEI)
if all_PG > 0:
    all_CFI = all_CEG / all_PG
else:
    print('error')
print(all_CEU)
print(all_CEG)
print(all_PL)
print(all_PG)
print(all_CEI)
print(all_CFI)
nownow = datetime.datetime.now()
print(nownow)
print ('province complete')

# AREA
# area_data_list = 0:area_id 1:area_num 2:area_type 3:parent_id 4:area_level 5:self_id
# node_result_list = 0:node_id 1:node_num 2:CEI 3:CEU 4:CEG 5:P_L 6:area_num
# 区域 江苏省 level3 、各地市 levle4、 各区县 level5
P = len(area_data_list) # 区域数量
A_Load = np.zeros(P, dtype = np.float) # 区域负荷
A_Cei = np.zeros(P, dtype = np.float)   # 区域平均碳排放因子
A_CUse = np.zeros(P, dtype = np.float)   # 区域用电碳排放量
A_CGen = np.zeros(P, dtype = np.float)   # 区域发电碳排放量
# cal level 5:
for node_item in node_result_list:
    for area_item in area_data_list:
        if node_item[6] == area_item[1]:
            A_Load[area_item[1]-1] = A_Load[area_item[1]-1] + node_item[5]
            A_CUse[area_item[1]-1] = A_CUse[area_item[1]-1] + node_item[3]
            A_CGen[area_item[1]-1] = A_CGen[area_item[1]-1] + node_item[4]
# cal level 4:
for area_item1 in area_data_list:
    for area_item2 in area_data_list:
        if int(area_item2[3]) == int(area_item1[5]) and int(area_item2[4]) == 5: #如果area_item2父区域为area_item1
            A_Load[area_item1[1]-1] = A_Load[area_item1[1]-1] + A_Load[area_item2[1]-1]
            A_CUse[area_item1[1]-1] = A_CUse[area_item1[1]-1] + A_CUse[area_item2[1]-1]
            A_CGen[area_item1[1]-1] = A_CGen[area_item1[1]-1] + A_CGen[area_item2[1]-1]
# cal level 3:
for area_item1 in area_data_list:
    for area_item2 in area_data_list:
        if int(area_item2[3]) == int(area_item1[5]) and int(area_item2[4]) == 4: #如果area_item2父区域为area_item1
            A_Load[area_item1[1]-1] = A_Load[area_item1[1]-1] + A_Load[area_item2[1]-1]
            A_CUse[area_item1[1]-1] = A_CUse[area_item1[1]-1] + A_CUse[area_item2[1]-1]
            A_CGen[area_item1[1]-1] = A_CGen[area_item1[1]-1] + A_CGen[area_item2[1]-1]
# cal level 2:
for area_item1 in area_data_list:
    for area_item2 in area_data_list:
        if int(area_item2[3]) == int(area_item1[5]) and int(area_item2[4]) == 3: #如果area_item2父区域为area_item1
            A_Load[area_item1[1]-1] = A_Load[area_item1[1]-1] + A_Load[area_item2[1]-1]
            A_CUse[area_item1[1]-1] = A_CUse[area_item1[1]-1] + A_CUse[area_item2[1]-1]
            A_CGen[area_item1[1]-1] = A_CGen[area_item1[1]-1] + A_CGen[area_item2[1]-1]
# cal level 1:
for area_item1 in area_data_list:
    for area_item2 in area_data_list:
        if int(area_item2[3]) == int(area_item1[5]) and int(area_item2[4]) == 2: #如果area_item2父区域为area_item1
            A_Load[area_item1[1]-1] = A_Load[area_item1[1]-1] + A_Load[area_item2[1]-1]
            A_CUse[area_item1[1]-1] = A_CUse[area_item1[1]-1] + A_CUse[area_item2[1]-1]
            A_CGen[area_item1[1]-1] = A_CGen[area_item1[1]-1] + A_CGen[area_item2[1]-1]
# cal area CEI
for i in range(0, P-1):
    if A_Load[i] > 0:
        A_Cei[i] = A_CUse[i] / A_Load[i]

# area_result_list = 0:area_id 1:Area_CEI 2:Area_DCEF 3:Area_ICEF
area_result_list = []
number = 0
for area_item in area_data_list:
    area_result_list.append([area_item[0], A_Cei[number], A_CGen[number], A_CUse[number], area_item[4], area_item[6]])
    number = number + 1

nownow = datetime.datetime.now()

print("carbon cal success")

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#                                                       输出结果插入数据库
#                     line_result_list = 0:line_id 1:line_num 2:Line_CEI 3:Line_CEF 4:Line_CEFR
#                     node_result_list = 0:node_id 1:node_num 2:CEI 3:CEU 4:CEG 5:P_L 6:area_num
#                     area_result_list = 0:area_id 1:Area_CEI 2:Area_DCEF 3:Area_ICEF
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
number = 1
table = o.get_table('hd_city_pro.self_ads_area_cal')
tunnel = TableTunnel(odps)
table.truncate()
upload_session = tunnel.create_upload_session(table.name)
with upload_session.open_record_writer(0) as writer:
    for area_item in area_result_list:
        record = table.new_record()
        record[0] = number
        record[1] = area_item[0]
        record[2] = nowdate
        record[3] = nowtime
        record[4] = area_item[1]
        record[5] = area_item[2]
        record[6] = area_item[3]
        writer.write(record)
        number = number + 1
upload_session.commit([0])

number = 1
table = o.get_table('hd_city_pro.self_ads_node_cal')
tunnel = TableTunnel(odps)
table.truncate()
upload_session = tunnel.create_upload_session(table.name)
with upload_session.open_record_writer(0) as writer:
    for node_item in node_result_list:
        record = table.new_record()
        record[0] = number
        record[1] = node_item[0]
        record[2] = nowdate
        record[3] = nowtime
        record[4] = node_item[2]
        record[5] = node_item[4]
        record[6] = node_item[3]
        record[7] = node_item[7]
        writer.write(record)
        number = number + 1
upload_session.commit([0])

number = 1
table = o.get_table('hd_city_pro.self_ads_line_cal')
tunnel = TableTunnel(odps)
table.truncate()
upload_session = tunnel.create_upload_session(table.name)
with upload_session.open_record_writer(0) as writer:
    for line_item in line_result_list:
        record = table.new_record()
        record[0] = number
        record[1] = line_item[0]
        record[2] = nowdate
        record[3] = nowtime
        record[4] = line_item[2]
        record[5] = line_item[4]
        record[6] = line_item[3]
        writer.write(record)
        number = number + 1
upload_session.commit([0])

nownow = datetime.datetime.now()
print(nownow)
print("insert result table success")