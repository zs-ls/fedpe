import random

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
import logging
import argparse
from config.get_config import get_config
from data.datasets import get_writer_num, FEMNIST
import matplotlib.pyplot as plt
from PIL import Image



class DataSet:
    def __init__(self, args):
        self.dataset = args.dataset
        self.distribution = args.data_distribution
        self.client_num = args.client_num
        self.train_samples_per_client = args.train_samples_per_client
        self.test_samples_per_client = args.test_samples_per_client
        self.train_set = None
        self.test_set = None
        self.all_clients_train_data = []
        self.all_clients_test_data = []
        self.all_clients_data = []
        self.classes_per_client = args.classes_per_client
        self.class_num = 0
        self.random_sample_writer = []
        self.train_data_rate = args.train_data_rate

    def load(self):
        if self.dataset == "MNIST":
            # print(self.dataset)
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
            self.test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
            self.class_num = len(self.train_set.classes)
        elif self.dataset == "FEMNIST":
            # print("hahahahahha")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            writer_nums = get_writer_num()
            # print(writer_nums)
            self.random_sample_writer = [
                np.random.choice(range(sum(writer_nums)), 1, replace=False).item()
                for _ in range(self.client_num)
            ]
            for i in range(self.client_num):
                client_dataset = FEMNIST(self.random_sample_writer[i], transform)
                client_dataset_size = len(client_dataset)
                # print(f"client_dataset_size: {client_dataset_size}")
                train_data_size = int(client_dataset_size * self.train_data_rate)
                test_data_size = client_dataset_size - train_data_size
                client_train_dataset, client_test_dataset = torch.utils.data.random_split(
                    client_dataset, [train_data_size, test_data_size]
                )
                self.all_clients_train_data.append(client_train_dataset)
                self.all_clients_test_data.append(client_test_dataset)
            self.class_num = 62
        return self

    def allocate(self):
        if self.dataset != 'FEMNIST':
            if self.distribution == "iid":
                self.load().random_iid_data_split()
            elif self.distribution == "non_iid":
                self.load().non_iid_data_split()
        else:
            self.load()
        return self.all_clients_train_data, self.all_clients_test_data

    def non_iid_data_split(self):
        logging.info("使用 non-iid 切分数据")
        # 首先将数据集按照类别划分
        train_indices_by_class = defaultdict(list)
        test_indices_by_class = defaultdict(list)

        for idx, (data, label) in enumerate(self.train_set):
            train_indices_by_class[label].append(idx)
        for idx, (data, label) in enumerate(self.test_set):
            test_indices_by_class[label].append(idx)

        # 按照类别打乱数据
        for label in train_indices_by_class:
            random.shuffle(train_indices_by_class[label])
        for label in test_indices_by_class:
            random.shuffle(test_indices_by_class[label])

        # 随机获取客户端的样本类别
        clients_classes = [
            np.random.choice(list(train_indices_by_class.keys()), self.classes_per_client, replace=False)
            for _ in range(self.client_num)
        ]
        # print(clients_classes)

        # 为每个客户端分配训练集和测试集
        for i, client_classes in enumerate(clients_classes):
            train_indices = []
            test_indices = []

            for cls in client_classes:
                train_samples_per_class = self.train_samples_per_client // self.classes_per_client
                test_samples_per_class = self.test_samples_per_client // self.classes_per_client
                train_indices.extend(train_indices_by_class[cls][:train_samples_per_class])
                test_indices.extend(test_indices_by_class[cls][:test_samples_per_class])
                train_indices_by_class[cls] = train_indices_by_class[cls][train_samples_per_class:]
                test_indices_by_class[cls] = test_indices_by_class[cls][test_samples_per_class:]
            train_data_set = torch.utils.data.Subset(self.train_set, train_indices)
            test_data_set = torch.utils.data.Subset(self.test_set, test_indices)
            self.all_clients_train_data.append(train_data_set)
            self.all_clients_test_data.append(test_data_set)
            # print("train set")
            # print(len(train_data_set))
            # number_class_train = {}
            # for data, label in train_data_set:
            #     if label not in number_class_train:
            #         number_class_train[label] = 0
            #     number_class_train[label] += 1
            # print(number_class_train)
            # print("test set")
            # print(len(test_data_set))
            # number_class_test = {}
            # for data, label in test_data_set:
            #     if label not in number_class_test:
            #         number_class_test[label] = 0
            #     number_class_test[label] += 1
            # print(number_class_test)

    def random_iid_data_split(self):
        logging.info("使用 iid 切分数据")
        train_indices = list(range(len(self.train_set)))
        test_indices = list(range(len(self.test_set)))

        random.shuffle(train_indices)
        random.shuffle(test_indices)

        for i in range(self.client_num):
            train_start_idx = i * self.train_samples_per_client
            train_end_idx = train_start_idx + self.train_samples_per_client
            test_start_idx = i * self.test_samples_per_client
            test_end_idx = test_start_idx + self.test_samples_per_client
            train_data_set = torch.utils.data.Subset(self.train_set, train_indices[train_start_idx:train_end_idx])
            test_data_set = torch.utils.data.Subset(self.test_set, test_indices[test_start_idx:test_end_idx])
            self.all_clients_train_data.append(train_data_set)
            self.all_clients_test_data.append(test_data_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config2.yaml', help='配置文件路径')
    command_args = parser.parse_args()

    # print("解析配置参数")
    args = get_config(command_args.config)
    dataset = DataSet(args)
    dataset.load().allocate()

    # train_samples_per_client = 200
    # test_samples_per_client = 50
    # all_clients_train_set, all_client_test_set = \
    #     DataSet("MINIST", "IID").allocate(client_num, train_samples_per_client, test_samples_per_client)
    #
    # 打印数据集长度（数据集包含多少样本）
    for i in range(dataset.client_num):
        train_set = dataset.all_clients_train_data[i]
        test_set = dataset.all_clients_test_data[i]
        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")

        # 获取一个样本图像和标签
        image, label = test_set[0]  # 获取第一个样本（图像，标签）
        # print(image)

        # 如果image是tensor，转换为numpy数组
        if torch.is_tensor(image):
            image = image.numpy()

        # 如果是PIL图像，转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 确保图像数据是2D的
        if len(image.shape) == 3:
            image = image.squeeze()  # 移除单维度

        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.show()

    exit()
    # a = {}
    # a[1] = 1
    # print(a)