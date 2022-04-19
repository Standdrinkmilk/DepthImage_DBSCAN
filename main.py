'''
Created on Feb 13, 2014

@author: sushant
'''

import csv
import os
import sys
import numpy as np

from tools.dbscanner import DBScanner


def get_data(config, DATA):
    data = []
    with open(DATA, 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        for id_, row in enumerate(csv_reader):
            if len(row) < config['dim']:
                print("ERROR: The data you have provided has fewer \
                    dimensions than expected (dim = %d < %d)"
                      % (config['dim'], len(row)))
                sys.exit()
            else:
                point = {'id': id_, 'value': []}
                for dim in range(0, config['dim']):
                    point['value'].append(float(row[dim]))
                data.append(point)
    return data


def read_config(CONFIG):
    config = {}
    try:
        with open(CONFIG, 'r') as file_obj:
            for line in file_obj:
                if line[0] != '#' and line.strip() != '':
                    key, value = line.split('=')
                    if '.' in value.strip():
                        config[key.strip()] = float(value.strip())
                    else:
                        config[key.strip()] = int(value.strip())
    except:
        print("Error reading the configuration file.\
            expected lines: param = value \n param = {eps, min_pts, dim}, \
            value = {float, int, int}")
        sys.exit()
    return config


def main():
    data_path = 'data/'
    config_path = 'config'
    path_list = os.listdir(data_path)
    # 先定义一个排序的空列表
    sort_num_list = []
    for file in path_list:
        sort_num_list.append(int(file.split('.csv')[0]))  # 去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
        sort_num_list.sort()  # 然后再重新排序

    # print(sort_num_list)
    # 接着再重新排序读取文件
    sorted_file = []

    for sort_num in sort_num_list:
        for file in path_list:
            if str(sort_num) == file.split('.csv')[0]:
                sorted_file.append(file)

    points_mean = []
    numbers = 0
    for file in sorted_file:
        print(file)
        csv_path = data_path + file
        config = read_config(config_path)
        dbc = DBScanner(config)
        data = get_data(config, csv_path)
        mean = dbc.dbscan(data)
        points_mean.append(mean)
        numbers += 1
        if numbers == 5:
            mean_array = np.array(points_mean)
            average_mean = np.mean(mean_array)
            average_var = np.var(mean_array)
            print('Average mean: {} Average var: {}'.format(average_mean, average_var))
            print('-------------------------------------------------------------')
            numbers = 0
            points_mean = []
        dbc.export()


if __name__ == "__main__":
    main()
