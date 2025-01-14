import copy
import time
import torch
import random
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt

from data.dataload import DataSet
from algorithms.FedPE.server import Server
from algorithms.FedPE.client import Client

logging.basicConfig(
    filename="./log/train.log",
    filemode="w",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


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
    start_time = time.time()
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
               copy.deepcopy(server.clients_net[i]), device, args.train_samples_per_client) for i in range(args.client_num)}

    logging.info("开始训练")
    round_cnt = 0
    all_mAP = []
    max_mAP = 0.0
    while True:
        logging.info("客户端采样")
        server.clients_sample()

        logging.info("将模型下发给选中的客户端，并开始本地训练")
        clients_param_list = []
        for i in server.clients_index_list:
            clients_param_list.append(clients[i].train(copy.deepcopy(server.clients_net[i])))

        logging.info("接收本地客户端发送的模型，并进行全局聚合")
        server.aggregation(clients_param_list)

        logging.info("评估本轮聚合后的模型")
        mAP = 0.0
        for i in range(len(clients_param_list)):
            mAP += clients_param_list[i]["mAP"]
            # print(clients_param_list[i]["mAP"])
        mAP /= len(clients_param_list)

        log_info = '    Round-{:>2d} | mAP:{:.2f}%'.format(
            round_cnt + 1, mAP * 100
        )
        logging.info(log_info)

        # if (float(test_auc) - args.object_acc) > 0:
        #     exit()
        all_mAP.append(mAP)
        max_mAP = max(max_mAP, mAP)
        round_cnt += 1
        if round_cnt >= 200:
            end_time = time.time()
            training_time = end_time - start_time
            # 将训练时间转换为小时、分钟、秒
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            # 使用logging输出训练时间
            logging.info(f"Training Time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")

            info = 'The best mAP is: {:.2f}%'.format(max_mAP * 100)
            logging.info(info)
            plt.plot(range(1, round_cnt + 1), all_mAP, label='FedPE', color='red')
            plt.xlabel('Communication Rounds')
            plt.ylabel('Test mAP')
            # plt.title('Test Accuracy vs Round')
            plt.legend()
            plt.grid(True)
            plt.show()
            exit()


