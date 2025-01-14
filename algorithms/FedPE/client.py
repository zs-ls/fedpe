import copy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging


class Client:
    def __init__(self, client_id, train_dataset, test_dataset, net, device, train_samples_per_client):
        self.client_id = client_id
        self.epoch = 10
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.net = net
        self.loss_func = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.data_nums = train_samples_per_client
        self.train_method = "fedpe"
        self.test_method = "mAP"
        self.pruning_param = 18.0
        self.expanding_param = 0.1
        self.accuracy_threshold = 0.5
        self.net_accuracy = 0.0
        self.learning_rate = 0.01
        self.batch_size = 5
        self.accumulate_prune_rate = 0.0
        self.max_prune_rate = 0.9
        self.residual = {name: torch.zeros_like(param).to(self.device) for name, param in self.net.named_parameters()}
        self.mask_dict = {name: torch.ones_like(param) for name, param in self.net.named_parameters() if "weight" in name}


    def train(self, global_net):
        if self.train_method == "basic":
            return self.basic_train_method(global_net)
        elif self.train_method == "fedpe":
            return self.fedpe_train_method(global_net)

    def fedpe_train_method(self, global_net):
        if global_net is not None:
            self.net = global_net

        client_param_dict = {
            "id": self.client_id,
            "net": {},
            "mask": {},
            "delta_accuracy": 0.0,
            "mAP": 0.0
        }

        self.net.to(self.device)
        # 对全局模型进行评估
        net_accuracy = self.test_net_as_map()
        # 计算模型准确率变化
        delta_accuracy = net_accuracy - self.net_accuracy
        logging.info("上轮准确率为：{:.2f}%，当前准确率为：{:.2f}%， 准确率变化值为：{:.2f}%，当前剪枝率为：{:.2f}%".format(
            self.net_accuracy * 100, net_accuracy * 100, delta_accuracy * 100, self.accumulate_prune_rate * 100))
        self.net_accuracy = net_accuracy
        client_param_dict["delta_accuracy"] = delta_accuracy
        accumulate_prune_rate = self.accumulate_prune_rate
        if delta_accuracy > 0:
            prune_rate = torch.sigmoid(torch.tensor(self.pruning_param * delta_accuracy)).item()
            accumulate_prune_rate += prune_rate
        elif delta_accuracy < 0:
            expand_rate = torch.sigmoid(torch.tensor(self.expanding_param * delta_accuracy)).item()
            accumulate_prune_rate -= expand_rate
        # 自适应剪枝-扩展模型
        if net_accuracy >= self.accuracy_threshold and delta_accuracy > 0 and accumulate_prune_rate <= self.max_prune_rate:
            net = copy.deepcopy(self.net)
            for name, param in self.net.named_parameters():
                if "weight" in name:
                    param.data.add_(self.residual[name].data)
            logging.info("对模型进行剪枝，剪枝率为：{:.2f}%，上轮剪枝率为：{:.2f}%，累计剪枝率为：{:.2f}%".format(
                prune_rate * 100, self.accumulate_prune_rate * 100, accumulate_prune_rate * 100))
            self.accumulate_prune_rate = accumulate_prune_rate
            self.unstructured_pruning(self.accumulate_prune_rate)
            for name, param in self.residual.items():
                if "weight" in name:
                    param.data.add_(net.state_dict()[name].data - self.net.state_dict()[name].data)
        elif delta_accuracy < 0 and accumulate_prune_rate > 0 and net_accuracy >= self.accuracy_threshold:
            net = copy.deepcopy(self.net)
            logging.info("对模型进行扩展，扩展率为：{:.2f}%，上轮剪枝率为：{:.2f}%，累计剪枝率为：{:.2f}%".format(
                expand_rate * 100, self.accumulate_prune_rate * 100, accumulate_prune_rate * 100))
            self.accumulate_prune_rate = accumulate_prune_rate
            self.expand(expand_rate)
            for name, param in self.residual.items():
                if "weight" in name:
                    param.data.add_(net.state_dict()[name].data - self.net.state_dict()[name].data)

        client_param_dict["mask"] = self.mask_dict

        self.net.train()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),  # 需要梯度计算的参数
            lr=self.learning_rate,  # 只设置学习率
        )

        train_data = DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True)
        loss_history = []
        for e in range(self.epoch):
            losses = []
            for b_idx, (images, labels) in enumerate(train_data):
                x = images.to(self.device)
                labels = labels.to(self.device)
                self.net.zero_grad()
                y = self.net(x)
                ce_loss = self.loss_func(y, labels)
                losses.append(ce_loss.item())
                ce_loss.backward()
                for name, param in self.net.named_parameters():
                    if "weight" in name:
                        param.grad *= self.mask_dict[name].to(self.device).float()
                optimizer.step()
            loss_history.append(np.mean(losses))
        loss = np.mean(loss_history)
        self.net.to("cpu")
        client_param_dict["net"] = self.net.state_dict()
        net_accuracy = self.test_net_as_map()
        client_param_dict["mAP"] = net_accuracy
        log_info = '\t Cli-{:>2d} \t | \t ceLoss:{:.6f} | \t acc:{:.2f}%'.format(self.client_id, loss, net_accuracy * 100)
        logging.info(log_info)

        return client_param_dict

    def basic_train_method(self, global_net):
        if global_net is not None:
            self.net = global_net

        client_param_dict = {
            "net": {},
            "sample_num": 200
        }

        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),  # 需要梯度计算的参数
            lr=self.learning_rate,  # 只设置学习率
        )

        train_data = DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True)
        loss_history = []
        for e in range(self.epoch):
            losses = []
            for b_idx, (images, labels) in enumerate(train_data):
                x = images.to(self.device)
                labels = labels.to(self.device)
                self.net.zero_grad()
                y = self.net(x)
                ce_loss = self.loss_func(y, labels)
                losses.append(ce_loss.item())
                ce_loss.backward()
                optimizer.step()
            loss_history.append(np.mean(losses))
        loss = np.mean(loss_history)
        self.net.to("cpu")
        client_param_dict["net"] = self.net.state_dict()
        log_info = '\t Cli-{:>2d} \t | \t ceLoss:{:.6f}'.format(self.client_id, loss)
        logging.info(log_info)

        return client_param_dict

    def unstructured_pruning(self, prune_rate):
        # self.net.to('cpu')
        for name, param in self.net.named_parameters():
            if "weight" in name:
                abs_param_data = torch.abs(param.data)
                threshold = torch.quantile(abs_param_data, prune_rate)
                mask = abs_param_data >= threshold
                self.mask_dict[name] = mask.to('cpu')
                param.data.mul_(mask.float())

    def expand(self, expand_rate):
        for name, param in self.net.named_parameters():
            if "weight" in name:
                abs_param_data = torch.abs(self.residual[name].data)
                threshold = torch.quantile(abs_param_data, 1.0 - expand_rate)
                mask = abs_param_data >= threshold
                self.mask_dict[name] |= mask.to('cpu')
                mask = mask.long()
                param.data[mask == 1] = self.residual[name].data[mask == 1]

    def test_net(self):
        if self.test_method == "mAP":
            return self.test_net_as_map()

    def test_net_as_map(self):
        self.net.to(self.device)
        self.net.eval()
        test_data = DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=False)
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

        return mean_average_precision


if __name__ == "__main__":
    print(torch.sigmoid(torch.tensor(18 * 0.025)))

