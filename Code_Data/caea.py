import pandas as pd
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
# line_data_list = line_data_list[line_data_list[' time '] == '20230611']
line_data_list[' start_nid '] = line_data_list[' start_nid '].apply(lambda x: float(x) if x!=' NULL      ' else -1)
line_data_list[' end_nid '] = line_data_list[' end_nid '].apply(
    lambda x: float(str(x)) if 'NULL' not in str(x) else -1)
line_data_list = line_data_list[[' line_id ',' data_date ',' time ',' line_p ',' start_nid ',' end_nid ',' line_name ']]
node_data_list = pd.merge(node_info, node_data, how='left', on=' node_num ')
node_data_list = node_data_list[
    [' node_id ', ' node_name ', ' node_vl ', ' node_type ', ' node_cei ', ' data_date ', ' time ', ' node_pg ', ' node_p ', ' area_num ']]
area_info_list = area_info[
    [' area_num ', ' area_type ', ' parent_id ', ' area_level ', ' area_id ', ' area_name ']]
node_data_list['zerp_list'] = [0]*len(node_data_list)
node_data_list = node_data_list.reset_index()[[' node_id ', 'index', ' node_type ', ' node_pg ', ' node_cei ', ' area_num ', ' node_p ', 'zerp_list']]
tmp_node_data_start = node_data_list[[' node_id ', 'index']]
tmp_node_data_start.columns = [' start_nid ', 'start_time']
tmp_node_data_end = node_data_list[[' node_id ', 'index']]
tmp_node_data_end.columns = [' end_nid ', 'end_time']
line_data_list = line_data_list[[' start_nid ', ' end_nid ', ' line_id ', ' line_p ']] #26万条
part_len = len(tmp_node_data_start)//20
for i in range(20):
    start_index = part_len * i
    if i == 19:
        end_index = len(tmp_node_data_start)
    else:
        end_index = part_len * (i+1)
    tmp_node_data_start_1 = tmp_node_data_start.iloc[start_index:end_index]
    tmp_node_data_start_1.columns = [' start_nid ', f'start_time_{i}']
    line_data_list = pd.merge(line_data_list, tmp_node_data_start_1, how='left', on=' start_nid ')
#line_data_list = pd.merge(line_data_list, tmp_node_data_start, how='left', on=' start_nid ')# 34990
# line_data_list = pd.merge(line_data_list, tmp_node_data_end, how='left', on=' end_nid ')
# line_data_list['number'] = list(range(1, len(line_data_list)+1))
# new_line_data_list = line_data_list[[' line_id ', 'number', 'start_time', 'end_time', ' line_p ']]

