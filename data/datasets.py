import json
import os.path
import sys
import gc
import re

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pympler import asizeof

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
all_data_dir = os.path.join(parent_path, 'dataset', 'FEMINIST', 'all_data')
data_by_writer_dir = os.path.join(parent_path, 'dataset', 'FEMINIST', 'raw', 'by_write')
user_name_list = []
max_user_id_in_json = [
    99,   199,  299,  399,  499,  599,
    699,  799,  899,  999,  1099, 1199,
    1299, 1399, 1499, 1601, 1701, 1801,
    1901, 2001, 2101, 2201, 2301, 2401,
    2501, 3101, 3202, 3302, 3402, 3502,
    3602, 3702, 3802, 3902, 4002, 4099
]

def get_writer_num():
    writers = []
    for dir in os.listdir(data_by_writer_dir):
        writer_dir = os.path.join(data_by_writer_dir, dir)
        writer_dir_list = os.listdir(writer_dir)
        writer_num = len(writer_dir_list)
        # print(writer_num)
        writers.append(writer_num)
        for file_name in writer_dir_list:
            user_name_list.append(file_name)
    # print(user_name_list)
    return writers


# 获取全部数据集，延迟拆分训练集和数据集
def get_user_data(user_ind):
    user_name = user_name_list[user_ind]
    user_id = int(re.search(r'\d+', str(user_name)).group(0))
    print(f'username: {user_name}, user_id: {user_id}')
    data = {'x': [], 'y': []}
    json_file_name = ''
    # 计算用户数据所在的json文件
    for i in range(len(max_user_id_in_json)):
        if user_id <= max_user_id_in_json[i]:
            json_file_name = f'all_data_{i}.json'
            # print(i)
            break
    file_path = os.path.join(all_data_dir, json_file_name)
    print(f"Loading {file_path}")
    with open(file_path, 'r') as file:
        file_data = json.load(file)
        for user, num_sample in zip(file_data['users'], file_data['num_samples']):
            if user_name != user:
                continue
            # print(f"{user_name} == {user}")
            sample_num = num_sample
            # print(f'sample_num: {sample_num}')
            for x_data, y_data in zip(file_data['user_data'][user]['x'], file_data['user_data'][user]['y']):
                data['x'].append(x_data)
                data['y'].append(y_data)
        del file_data
    print("Loading completed")
    return data

# 直接获取数据的时候切分训练集和数据集
# def get_user_data(user_ind, train_size_rate):
#     user_name = user_name_list[user_ind]
#     user_id = int(re.search(r'\d+', str(user_name)).group(0))
#     print(f'username: {user_name}, user_id: {user_id}')
#     data = {'train_sample': {'x': [], 'y': []}, 'test_sample': {'x': [], 'y': []}}
#     json_file_name = ''
#     # 计算用户数据所在的json文件
#     for i in range(len(max_user_id_in_json)):
#         if user_id <= max_user_id_in_json[i]:
#             json_file_name = f'all_data_{i}.json'
#             print(i)
#             break
#     file_path = os.path.join(all_data_dir, json_file_name)
#     print(f"Loading {file_path}")
#     with open(file_path, 'r') as file:
#         file_data = json.load(file)
#         for user, num_sample in zip(file_data['users'], file_data['num_samples']):
#             if user_name != user:
#                 continue
#             print(f"{user_name} == {user}")
#             sample_num = num_sample
#             print(f'sample_num: {sample_num}')
#             train_data_num = int(sample_num * train_size_rate)
#             print(f'train_data_num: {train_data_num}')
#             for idx, (x_data, y_data) in enumerate(zip(file_data['user_data'][user]['x'], file_data['user_data'][user]['y'])):
#                 if idx <= train_data_num:
#                     data['train_sample']['x'].append(x_data)
#                     data['train_sample']['y'].append(y_data)
#                 else:
#                     data['test_sample']['x'].append(x_data)
#                     data['test_sample']['y'].append(y_data)
#     # print(f"data: {data}")
#     return data


    # file_cnt = 0
    # for filename in os.listdir(all_data_dir):
    #     if filename.endswith(".json"):
    #         file_path = os.path.join(all_data_dir, filename)
    #         file_cnt += 1
    #         print(f"Loading {file_path}")
    #
    #         for users, num_samples, file_data in process_file(file_path):
    #             # 合并文件数据
    #             for user, num_sample in zip(users, num_samples):
    #                 data['users'].append(user)
    #                 data['num_samples'].append(num_sample)
    #
    #             for user, user_data in file_data.items():
    #                 if user not in data['user_data']:
    #                     data['user_data'][user] = {'x': [], 'y': []}
    #                 for x, y in zip(user_data['x'], user_data['y']):
    #                     data['user_data'][user]['x'].append(x)
    #                     data['user_data'][user]['y'].append(y)
    #
    #         del users, num_samples, file_data
    #         gc.collect()
    #
    #
    # print(f"{file_cnt} datafiles loading finished")
    # print(sys.getsizeof(data))
    # return data


class FEMNIST(Dataset):
    # 直接获取数据的时候切分训练集和数据集
    # def __init__(self, user_ind, train_data_rate, is_train=True, transform=None):
    #     self.is_train = is_train
    #     self.data = get_user_data(user_ind, train_data_rate)
    #     self.transform = transform
    #
    #     if self.is_train:
    #         self.samples = self.data['train_samples']
    #     else:
    #         self.samples = self.data['test_samples']

    # 获取全部数据集，延迟拆分训练集和数据集
    def __init__(self, user_ind, transform=None):
        self.data = get_user_data(user_ind)
        self.transform = transform

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, item):
        x_data = self.data['x'][item]
        y_data = self.data['y'][item]
        # 确保x_data为numpy数组
        if isinstance(x_data, list):
            x_data = np.array(x_data)

        # 重要：确保数据范围在0-255之间
        x_data = x_data.reshape(28, 28)  # 重塑为28x28
        x_data = (x_data * 255).astype(np.uint8)  # 转换为0-255范围的uint8类型

        # 转换为PIL图像
        x_data = Image.fromarray(x_data, mode="L")

        # 应用transform（如果有）
        if self.transform:
            x_data = self.transform(x_data)
        return x_data, y_data


if __name__ == "__main__":
    print(get_writer_num())
    # data = get_dataset()
    # show_images(data[0]["inputs"], data[0]["labels"])
