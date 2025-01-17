import copy
import yaml

import torch
import numpy as np
import sys
import torch.nn as nn

sys.path.append("../../")
from models.lenet import LeNet
from models.cnn import CNN
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
        self.algorithm = args.algorithm
        self.clients_delta_accuracy = []
        self.client_sample_method = args.client_sample_method
        self.clients_net = []
        self.clients_mask = []
        self.batch_size = args.batch_size
        self.class_num = 0
        self.model_parameters_num = 0

    def model_init(self, class_num):
        print(class_num)
        self.class_num = class_num
        if self.model_type == "LeNet":
            self.net = LeNet(self.class_num)
        elif self.model_type == "CNN":
            self.net = CNN(self.class_num)
        for name, param in self.net.named_parameters():
            if "weight" in name:
                self.model_parameters_num += param.numel()
        for i in range(self.client_num):
            self.clients_mask.append({
                name: torch.ones_like(param, dtype=bool)
                for name, param in self.net.named_parameters()
                if "weight" in name
            })
            self.clients_net.append(copy.deepcopy(self.net))
            self.clients_delta_accuracy.append(0.0)
                # for name, param in self.clients_net[i].named_parameters():
                #     if "weight" in name:
                #         param.data.mul_(self.clients_mask[i][name].float())

    def clients_sample(self):
        # logging.info(""开始采样喽)
        if self.client_sample_method == "random":
            self.random_client_sample()

    def aggregation(self, clients_param_list):
        if self.algorithm == "FedAvg":
            self.avg(clients_param_list)
        elif self.algorithm == "FedPE":
            self.fair(clients_param_list)

    def fair(self, clients_param_list):
        clients_delta_accuracy_sum = 0.0

        for client_param in clients_param_list:
            client_id = client_param["id"]
            self.clients_net[client_id] = client_param["net"]
            self.clients_mask[client_id] = client_param["mask"]
            client_delta_accuracy = torch.sigmoid(torch.tensor(client_param["delta_accuracy"])).item()
            self.clients_delta_accuracy[client_id] = client_delta_accuracy
            clients_delta_accuracy_sum += client_delta_accuracy


        # aggregated_weight = [1.0 * self.clients_delta_accuracy[client_id] / clients_delta_accuracy_sum
        #                      for client_id in self.clients_index_list]

        aggregated_net = copy.deepcopy(self.net)
        flag = 0
        for name, param in aggregated_net.named_parameters():
            param.data.fill_(0.0)
            aggregated_weight = torch.zeros_like(param)
            if "weight" in name:
                for i in self.clients_index_list:
                    aggregated_weight.add_(self.clients_mask[i][name].long() * self.clients_delta_accuracy[i])
            for i in self.clients_index_list:
                if "weight" in name:
                    param.data.add_(torch.where(
                        aggregated_weight != 0,
                        self.clients_net[i][name] * self.clients_delta_accuracy[i] / aggregated_weight,
                        torch.zeros_like(param)
                    ))
                    # if flag == 0:
                    #     print("输出客户端的掩码")
                    #     print(self.clients_mask[i][name].long())
                    #     print("输出加权后的模型参数")
                    #     print(param)
                        # print("输出每个元素的权值")
                        # print(self.clients_delta_accuracy[i] / aggregated_weight)
                    flag = 1
                else:
                    param.data.add_(self.clients_net[i][name] * self.clients_delta_accuracy[i] / clients_delta_accuracy_sum)

        self.net = aggregated_net
        for i in range(self.client_num):
            client_net = copy.deepcopy(self.net)
            for name, param in client_net.named_parameters():
                if "weight" in name:
                    param.data.mul_(self.clients_mask[i][name].float())
                    # if flag == 0:
                    #     print("输出客户端上传的模型参数")
                    #     print(param.data)
                    #     print("输出客户端的掩码")
                    #     print(self.clients_mask[i][name].long())
                    #     print("输出模型参数的是否参与梯度更新")
                    #     print(param.requires_grad)
                    #     print("与掩码进行运算后，输出模型参数是否参与梯度更新")
                    #     param.requires_grad = self.clients_mask[i][name]
                    #     print(param.requires_grad)
                    #     flag = 1
            self.clients_net[i] = client_net

    def avg(self, clients_param_list):
        clients_net = []
        clients_train_data_num = []
        for client_param in clients_param_list:
            clients_net.append(client_param["net"])
            clients_train_data_num.append(client_param["samples"])

        clients_train_data_sum = sum(clients_train_data_num)
        aggregated_weight = [1.0 * client_data_num / clients_train_data_sum
                             for client_data_num in clients_train_data_num]
        print(aggregated_weight)
        aggregated_net = self.net.state_dict()
        for key in aggregated_net.keys():
            aggregated_net[key] = torch.zeros_like(aggregated_net[key])
            for i in range(len(clients_net)):
                aggregated_net[key] += aggregated_weight[i] * clients_net[i][key]

        self.net.load_state_dict(aggregated_net)
        for i in range(self.client_num):
            client_net = copy.deepcopy(self.net)
            for name, param in client_net.named_parameters():
                if "weight" in name:
                    # print(self.clients_mask[i][name])
                    param.data.mul_(self.clients_mask[i][name].float())
            self.clients_net[i] = client_net

    def random_client_sample(self):
        sample_nums = max(int(self.client_sample_rate * self.client_num), 1)
        all_client_idx = [i for i in range(self.client_num)]
        self.clients_index_list = list(np.random.choice(all_client_idx, sample_nums, replace=False))

    def test_net(self, test_set):
        self.net.to(self.device)
        self.net.eval()
        correct = 0
        test_data = DataLoader(test_set, batch_size=5,
                                 shuffle=False, drop_last=False)
        loss_func = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        losses = []

        with torch.no_grad():
            for _, (data, target) in enumerate(test_data):
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
        accuracy = 100.0 * correct / len(test_data.dataset)
        self.net.to("cpu")
        return '%.3f' % accuracy, '%.3f' % loss

    def test_net_as_map(self, test_set):
        self.net.to(self.device)
        self.net.eval()
        test_data = DataLoader(test_set, self.batch_size, shuffle=False, drop_last=False)
        # 用于存储每个类别的统计
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for _, (data, labels) in enumerate(test_data):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.net(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = outputs.data.max(1, keepdim=True)[1]

                # 遍历当前批次中的样本
                for label, pred in zip(labels, preds):
                    label = label.item()
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1  # 累计该类别的样本数
                    if label == pred.item():
                        class_correct[label] += 1  # 累计正确预测的样本数

        # 计算加权类别准确率
        total_classes = len(class_total)  # 所有类别的总样本数
        average_precision = sum(class_correct[c] / class_total[c] for c in class_total)
        mean_average_precision = average_precision / total_classes
        self.net.to("cpu")

        return mean_average_precision


import os
if __name__ == "__main__":
    print(torch.sigmoid(torch.tensor(18*0.1)))
    print(", ".join(["1.1", "2.1"]))
    # pro_name = "test"
    # result_file = f"./result/{pro_name}.xlsx"
    # directory = os.path.dirname(result_file)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)  # 创建父目录
    #     print(f"目录不存在，已创建: {directory}")
    #     open(result_file, 'w').close()
    # all_args = get_config("../../config/config.yaml")
    # args = all_args.train_args
    # device = torch.device('cuda:{}'.format(
    #     args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # server = Server(args, device)
    # server.model_init()
    # server.unstructured_pruning(0.2)
    # x = torch.randn(3, 4)
    # print(x)
    # abs_x = torch.abs(x)
    # q = 0.25
    # threshold = torch.quantile(abs_x, q)
    # print(threshold)
    # mask = abs_x >= threshold
    # print(mask)
    # x = x * mask.float()
    # print(x)
    # list = [1, 3, 3, 2, 1, 5, 6, 7, 8, 9]
    # ind_list = np.random.choice(list, 5, replace=True)
    # print(ind_list)


