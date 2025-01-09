import copy

import torch
import random
import numpy as np
import yaml
import logging

from data.dataload import DataSet
from algorithms.FedPE.server import Server
from algorithms.FedPE.client import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_config(path):
    with open(path, 'r') as file:
        config_data = yaml.safe_load(file)
        return Config(config_data)


class Config:
    def __init__(self, args):
        # 将传入的数据转换为属性
        if isinstance(args, dict):
            for key, value in args.items():
                # 遍历字典项并递归转换
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
        else:
            self.args = args

    def __repr__(self):
        return f"Config({self.__dict__})"


if __name__ == "__main__":
    logging.info("设置随机种子")
    setup_seed(0)

    logging.info("解析配置参数")
    all_args = get_config("./config/config.yaml")
    args = all_args.train_args

    logging.info("选择进行训练的设备")
    device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    logging.info("为每个客户端分配数据")
    dataset = DataSet("MINIST", "IID")
    all_clients_train_set, all_client_test_set = \
        dataset.allocate(args.client_num, args.train_samples_per_client, args.test_samples_per_client)

    logging.info("初始化服务器端")
    server = Server(args, device)

    logging.info("初始化全局模型")
    server.model_init()

    logging.info("初始化客户端")
    clients = {i: Client(i, all_clients_train_set[i], all_client_test_set[i],
               copy.deepcopy(server.net), device, args.train_samples_per_client) for i in range(args.client_num)}

    logging.info("开始训练")
    round_cnt = 0
    while True:
        logging.info("客户端采样")
        server.clients_sample()

        logging.info("将模型下发给选中的客户端，并开始本地训练")
        clients_param_list = []
        for i in server.clients_index_list:
            clients_param_list.append(clients[i].train(copy.deepcopy(server.net)))

        logging.info("接收本地客户端发送的模型，并进行全局聚合")
        server.aggregation(clients_param_list)

        logging.info("评估本轮聚合后的模型")
        test_auc, test_loss = server.test_net(dataset.test_set)

        log_info = '    Round-{:>2d} | Loss: ce:{} | Acc: test:{}%'.format(
            round_cnt + 1, test_loss, test_auc
        )
        logging.info(log_info)

        if (float(test_auc) - args.object_acc) > 0:
            exit()

        round_cnt += 1


