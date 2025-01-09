import copy
import logging
import yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import numpy as np
import sys
import torch.nn as nn

sys.path.append("../../")
from models.lenet import LeNet
from torch.utils.data import DataLoader


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


class Server:
    def __init__(self, args, device):
        self.device = device
        self.client_num = args.client_num
        self.client_sample_rate = args.client_sample_rate
        self.clients_index_list = None
        self.model_type = args.model_type
        self.net = None
        self.aggregated_method = args.aggregated_method
        self.client_sample_method = args.client_sample_method

    def model_init(self):
        if self.model_type == "LeNet":
            self.net = LeNet(10)

    def clients_sample(self):
        # logging.info(""开始采样喽)
        if self.client_sample_method == "random":
            self.random_client_sample()

    def aggregation(self, clients_param_dict):
        if self.aggregated_method == "FedAvg":
            self.avg(clients_param_dict)

    def unstructured_pruning(self, prune_rate):
        pruning_net = copy.deepcopy(self.net)
        for name, param in pruning_net.named_parameters():
            if "weight" in name:
                print(name)

    def test_net(self, test_set):
        self.net.to(self.device)
        self.net.eval()
        correct = 0
        data_loader = DataLoader(test_set, batch_size=5,
                                 shuffle=False, drop_last=False)
        loss_func = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        losses = []

        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                if isinstance(data, tuple):
                    data = data[-1]
                data = data.to(self.device)
                target = target.to(self.device)
                log_probs = self.net(data)
                batch_loss = loss_func(log_probs, target).item()
                losses.append(batch_loss)
                if isinstance(log_probs, tuple):
                    log_probs = log_probs[0]
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)
                                     ).long().cpu().sum()
        loss = sum(losses) / max(len(losses), 1)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        self.net.to("cpu")
        return '%.3f' % accuracy, '%.3f' % loss

    def avg(self, clients_param_list):
        clients_net = []
        clients_train_data_num = []
        for client_param in clients_param_list:
            clients_net.append(client_param["net"])
            clients_train_data_num.append(client_param["sample_num"])

        clients_train_data_sum = sum(clients_train_data_num)
        aggregated_weight = [1.0 * client_data_num / clients_train_data_sum
                             for client_data_num in clients_train_data_num]
        aggregated_net = self.net.state_dict()
        for key in aggregated_net.keys():
            aggregated_net[key] = torch.zeros_like(aggregated_net[key])
            for i in range(len(clients_net)):
                aggregated_net[key] += aggregated_weight[i] * clients_net[i][key]

        self.net.load_state_dict(aggregated_net)

    def random_client_sample(self):
        sample_nums = max(int(self.client_sample_rate * self.client_num), 1)
        all_client_idx = [i for i in range(self.client_num)]
        self.clients_index_list = list(np.random.choice(all_client_idx, sample_nums, replace=False))


if __name__ == "__main__":
    all_args = get_config("../../config/config.yaml")
    args = all_args.train_args
    device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    server = Server(args, device)
    server.model_init()
    server.unstructured_pruning(0.2)
