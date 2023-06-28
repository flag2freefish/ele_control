from functools import wraps
import time
import pandas as pd
import datetime


# time装饰器
def timer(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        begin_time = time.perf_counter()
        result = func(*args, **kwargs)
        start_time = time.perf_counter()
        print('func:%r took: %2.4f sec' % (func.__name__, start_time - begin_time))
        return result

    return wrap
def read_file():
    line_data = pd.read_csv("line_data.txt", sep='\t')
    line_info = pd.read_csv("line_info.txt", sep='\t')
    node_data = pd.read_csv("node_data.txt", sep='\t')
    node_info = pd.read_csv("node_info.txt", sep='\t')
    area_info = pd.read_csv("area_info.txt", sep='\t')
    line_data_list = pd.merge(line_info, line_data, how='left', on=' line_num ')
    # line_data_list[' line_vl '] = line_data_list[' line_vl '].apply(lambda x: str(x))
    # line_data_list = line_data_list[line_data_list[' line_vl '] >= '110']
    # line_data_list[' data_date '] = line_data_list[' data_date '].apply(lambda x: str(x))
    # line_data_list = line_data_list[line_data_list[' data_date '] == '20230611']
    # line_data_list = line_data_list[line_data_list[' time '] == '10000']
    line_data_list[' start_nid '] = line_data_list[' start_nid '].apply(lambda x: float(x) if x!=' NULL      ' else -1)
    line_data_list[' end_nid '] = line_data_list[' end_nid '].apply(
        lambda x: float(str(x)) if 'NULL' not in str(x) else -1)
    line_data_list = line_data_list[[' line_id ',' data_date ',' time ',' line_p ',' start_nid ',' end_nid ',' line_name ']].values.tolist()
    node_data_list = pd.merge(node_info, node_data, how='left', on=' node_num ')
    # node_data_list[' node_vl '] = node_data_list[' node_vl '].apply(lambda x: str(x))
    # node_data_list[' data_date '] = node_data_list[' data_date '].apply(lambda x: str(x))
    # node_data_list = node_data_list[node_data_list[' node_vl '] >= '110']
    # node_data_list = node_data_list[node_data_list[' data_date '] == '20230611']
    # node_data_list = node_data_list[node_data_list[' time '] == '100000']
    node_data_list = node_data_list[
        [' node_id ', ' node_name ', ' node_vl ', ' node_type ', ' node_cei ', ' data_date ', ' time ', ' node_pg ', ' node_p ', ' area_num ']].values.tolist()
    # area_info[' area_level '] = area_info[' area_level '].apply(lambda x: str(x))
    # area_info_list = area_info[area_info[' area_level '] >= '0']
    # area_info_list = area_info_list[
    area_info_list = area_info[
        [' area_num ', ' area_type ', ' parent_id ', ' area_level ', ' area_id ', ' area_name ']].values.tolist()
    return line_data_list, node_data_list, area_info_list

@timer
def old_node_data_list(node_db_list):
    # node_data_list数据生成
    node_data_list = []
    number = 1
    for index in node_db_list:
        temp = [index[0], number, index[3], index[7], index[4], index[9], index[8], 0]
        node_data_list.append(temp)
        # node_data_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag
        number = number + 1
    return node_data_list

@timer
def new_node_data_list(node_db_list):
    # node_data_list数据生成
    b = range(1, len(node_db_list) + 1)
    return list(map(lambda x, y: [x[0], y, x[3], x[7], x[4], x[9], x[8], 0], node_db_list, b))

@timer
def old_line_data_list(line_db_list, node_data_list):
    # line_data_list数据生成
    # node_cal_list为剔除空节点后的节点数据列表
    node_cal_list = []
    number = 1
    for node_data_item in node_data_list:
        for index in line_db_list:
            if index[4] == node_data_item[0]:
                if node_data_item not in node_cal_list:
                    node_cal_list.append(node_data_item)
            if index[5] == node_data_item[0]:
                if node_data_item not in node_cal_list:
                    node_cal_list.append(node_data_item)
    # 节点重新编号
    number = 1
    for index in node_cal_list:
        index[1] = number
        number = number + 1
        # node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag
    return node_cal_list

@timer
def new_line_data_list(line_db_list, node_data_list):
    # line_data_list数据生成
    # node_cal_list为剔除空节点后的节点数据列表
    target_line_1 = list(map(lambda x: x[4], line_db_list))
    target_line_2 = list(map(lambda x: x[5], line_db_list))
    node_cal_list = []
    for node_data_item in node_data_list:
        if node_data_item[0] in target_line_1 or node_data_item[0] in target_line_2:
            node_cal_list.append(node_data_item)
    # 节点重排号
    b = range(1, len(node_cal_list) + 1)
    def replace(x, y):
        x[1] = y
        return x
    return list(map(lambda x, y: replace(x,y), node_cal_list, b))

@timer
def old_line_code(line_db_list, node_cal_list):
    # 线路重新编号
    line_data_list = []
    number = 1
    for index in line_db_list:
        start_num = None
        end_num = None
        for node_data_item in node_cal_list:
            if index[4] == node_data_item[0]:
                start_num = node_data_item[1]
            if index[5] == node_data_item[0]:
                end_num = node_data_item[1]
            # print (index[4])
            # print (node_data_item[0])
        temp = [index[0], number, start_num, end_num, index[3]]
        line_data_list.append(temp)
        # line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
        number = number + 1
    return line_data_list

@timer
def new_line_code(line_db_list, node_cal_list):
    # 线路重新编号
    node_data_item_0 = list(map(lambda x: x[0], node_cal_list))
    node_data_item_1 = list(map(lambda x: x[1], node_cal_list))
    b = list(range(1, len(line_db_list) + 1))
    def x_y(x, y):
        if x[4] in node_data_item_0:
            return [x[0], y, node_data_item_1[node_data_item_0.index(x[4])], node_data_item_1[node_data_item_0.index(x[5])], x[3]]
    return list(map(lambda x, y:x_y(x, y), line_db_list, b))
    #return list(map(lambda x, y: [x[0], y, node_data_item_1[node_data_item_0.index(x[4])], node_data_item_1[node_data_item_0.index(x[5])], x[3]], line_db_list, b))

@timer
def old_area_code(area_db_list, node_cal_list):
    area_data_list = []
    number = 1
    for index in area_db_list:
        area_data_list.append([index[0], number, index[1], index[2], int(index[3]), index[4], index[5]])
        number = number + 1

    for node_item in node_cal_list:
        for area_item in area_data_list:
            if node_item[5] == area_item[0]:
                node_item[5] = area_item[1]
    return area_data_list


@timer
def new_area_code(area_db_list, node_cal_list):
    b = range(1, len(area_db_list) + 1)
    area_data_list = list(map(lambda x, y: [x[0], y, x[1], x[2], int(x[3]), x[4], x[5]], area_db_list, b))

    area_data_item_0 = list(map(lambda x: x[0], area_data_list))
    area_data_item_1 = list(map(lambda x: x[1], area_data_list))
    def replace(x):
        x[5] = area_data_item_1[area_data_item_0.index(x[5])]
        return x
    node_cal_list = list(map(lambda x: replace(x), node_cal_list))
    return area_data_list, node_cal_list

if __name__ == "__main__":
    line_db_list, node_db_list, area_db_list = read_file()
    node_data_list_1 = old_node_data_list(node_db_list)
    node_data_list_2 = new_node_data_list(node_db_list)
    print(node_data_list_2==node_data_list_1)
    filename = open('node_data_list_1.txt', 'w')
    for value in node_data_list_1:
        filename.write(str(value))
    filename.close()
    filename = open('node_data_list_2.txt', 'w')
    for value in node_data_list_2:
        filename.write(str(value))
    filename.close()
    # node_cal_list_1 = old_line_data_list(line_db_list, node_data_list_1)
    node_cal_list_2 = new_line_data_list(line_db_list, node_data_list_2)
    # print(node_cal_list_1 == node_cal_list_2)
    # filename = open('node_cal_list_1.txt', 'w')
    # for value in node_cal_list_1:
    #     filename.write(str(value))
    # filename.close()
    filename = open('node_cal_list_2.txt', 'w')
    for value in node_cal_list_2:
        filename.write(str(value))
    filename.close()
    # f = open('node_cal_list_1.txt', 'r')
    # node_cal_list_1 = eval(f.read())
    # f = open('node_cal_list_2.txt', 'r')
    # node_cal_list_2 = eval(f.read())
    # line_data_list_1 = old_line_code(line_db_list, node_cal_list_1)
    line_data_list_2 = new_line_code(line_db_list, node_cal_list_2)
    # print(line_data_list_1 == line_data_list_2)
    # filename = open('line_data_list_1.txt', 'w')
    # for value in line_data_list_1:
    #     filename.write(str(value))
    # filename.close()
    filename = open('line_data_list_2.txt', 'w')
    for value in line_data_list_2:
        filename.write(str(value))
    filename.close()
    # area_data_list_1, node_cal_list_3 = old_area_code(area_db_list, node_cal_list_1)
    area_data_list_2, node_cal_list_4 = new_area_code(area_db_list, node_cal_list_2)
    # print(area_data_list_1 == area_data_list_2)
    # print(node_cal_list_3 == node_cal_list_4)
    # filename = open('area_data_list_1.txt', 'w')
    # for value in area_data_list_1:
    #     filename.write(str(value))
    # filename.close()
    # filename = open('node_cal_list_3.txt', 'w')
    # for value in node_cal_list_3:
    #     filename.write(str(value))
    # filename.close()
    # filename = open('area_data_list_2.txt', 'w')
    # for value in area_data_list_2:
    #     filename.write(str(value))
    # filename.close()
    # filename = open('node_cal_list_4.txt', 'w')
    # for value in node_cal_list_4:
    #     filename.write(str(value))
    # filename.close()


