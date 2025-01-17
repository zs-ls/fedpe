import copy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging


class Client:
    def __init__(self, client_id, train_dataset, test_dataset, net, device, args):
        self.client_id = client_id
        self.epoch = args.local_epoch
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.net = net
        self.loss_func = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.data_nums = len(train_dataset)
        self.algorithm = args.algorithm
        self.pruning_param = args.pruning_param
        self.expanding_param = args.expanding_param
        self.accuracy_threshold = args.accuracy_threshold
        self.net_accuracy = 0.0
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.accumulate_prune_rate = 0.0
        self.max_prune_rate = args.max_prune_rate
        self.compression_ratio = 1.0
        self.residual = {name: torch.zeros_like(param).to(self.device) for name, param in self.net.named_parameters()}
        self.mask_dict = {name: torch.ones_like(param) for name, param in self.net.named_parameters() if "weight" in name}
        self.prune_flag = False

    def train(self, round, global_net):
        if self.algorithm == "FedAvg":
            return self.fedavg_train_method(round, global_net)
        elif self.algorithm == "FedPE":
            return self.fedpe_train_method(round, global_net)

    def fedpe_train_method(self, round, global_net):
        # print(f"本地{self.algorithm}训练")
        # if round > 1:
        #     self.prune_flag = True
        if global_net is not None:
            self.net = global_net

        client_param_dict = {
            "id": self.client_id,
            "net": {},
            "mask": {},
            "delta_accuracy": 0.0,
            "mAP": 0.0,
            "CR": 0.0,
            "valid_param_num": 0
        }

        self.net.to(self.device)
        # 对全局模型进行评估
        net_accuracy = self.test_net_as_map()
        # 计算模型准确率变化
        delta_accuracy = net_accuracy - self.net_accuracy
        logging.info("上轮准确率为：{:.2f}%，当前准确率为：{:.2f}%， 准确率变化值为：{:.2f}%，当前剪枝率为：{:.2f}%".format(
            self.net_accuracy * 100, net_accuracy * 100, delta_accuracy * 100, self.accumulate_prune_rate * 100))
        self.net_accuracy = net_accuracy
        # print("{} - {}".format(self.client_id, self.net_accuracy))
        client_param_dict["delta_accuracy"] = delta_accuracy
        accumulate_prune_rate = self.accumulate_prune_rate
        prune_rate = 0.0
        expand_rate = 0.0
        if delta_accuracy > 0:
            prune_rate = torch.sigmoid(torch.tensor(self.pruning_param * delta_accuracy)).item()
            accumulate_prune_rate += prune_rate * (1 - accumulate_prune_rate)
        elif delta_accuracy < 0:
            expand_rate = torch.sigmoid(torch.tensor(self.expanding_param * delta_accuracy)).item()
            accumulate_prune_rate -= expand_rate
        # 自适应剪枝-扩展模型
        if self.prune_flag:
            if (net_accuracy >= self.accuracy_threshold and 0 < delta_accuracy <= 0.1 and
                    self.accumulate_prune_rate <= self.max_prune_rate):
                # if accumulate_prune_rate < 1.0:
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
            elif delta_accuracy < 0:
                if expand_rate <= self.accumulate_prune_rate:
                    net = copy.deepcopy(self.net)
                    logging.info("对模型进行扩展，扩展率为：{:.2f}%，上轮剪枝率为：{:.2f}%，累计剪枝率为：{:.2f}%".format(
                        expand_rate * 100, self.accumulate_prune_rate * 100, accumulate_prune_rate * 100))
                    self.accumulate_prune_rate = accumulate_prune_rate
                    self.expand(expand_rate)
                    for name, param in self.residual.items():
                        if "weight" in name:
                            param.data.add_(net.state_dict()[name].data - self.net.state_dict()[name].data)

        client_param_dict["mask"] = self.mask_dict
        # 计算本轮相对FedAvg的压缩率
        param_sum = 0
        valid_param = 0
        for name, param in self.net.named_parameters():
            if "weight" in name:
                param_sum += param.numel()
                # print((self.mask_dict[name].int() == 1).sum().item())
                # exit()
                valid_param += (self.mask_dict[name].int() == 1).sum().item()
        self.compression_ratio = 1.0 * param_sum / valid_param
        # print("模型参数数量：{} | 有效参数量：{}".format(param_sum, valid_param))
        client_param_dict["CR"] = self.compression_ratio
        client_param_dict["valid_param_num"] = valid_param

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
        log_info = '\t Round-{:>2d} | \t Cli-{:>2d} \t | \t ceLoss:{:.6f} | \t acc:{:.2f}%'.format(
            round + 1, self.client_id, loss, net_accuracy * 100
        )
        logging.info(log_info)

        return client_param_dict

    def fedavg_train_method(self, round, global_net):
        # print(f"本地{self.algorithm}训练")
        if global_net is not None:
            self.net = global_net

        client_param_dict = {
            "id": self.client_id,
            "net": {},
            "mAP": 0.0,
            "CR": 1.0,
            "valid_param_num": 0,
            "samples": self.data_nums
        }

        param_sum = 0
        valid_param = 0
        for name, param in self.net.named_parameters():
            if "weight" in name:
                param_sum += param.numel()
                # print((self.mask_dict[name].int() == 1).sum().item())
                # exit()
                valid_param += (self.mask_dict[name].int() == 1).sum().item()
        self.compression_ratio = 1.0 * param_sum / valid_param
        # print("模型参数数量：{} | 有效参数量：{}".format(param_sum, valid_param))
        client_param_dict["CR"] = self.compression_ratio
        client_param_dict["valid_param_num"] = valid_param

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
                optimizer.zero_grad()
                y = self.net(x)
                ce_loss = self.loss_func(y, labels)
                losses.append(ce_loss.item())
                ce_loss.backward()
                for name, param in self.net.named_parameters():
                    if "weight" in name:
                        param.grad *= self.mask_dict[name].to(self.device).float()
                        # print(param.grad)
                optimizer.step()
            loss_history.append(np.mean(losses))
        loss = np.mean(loss_history)
        self.net.to("cpu")
        client_param_dict["net"] = self.net.state_dict()
        # net_accuracy = self.test_net_as_map()
        # client_param_dict["mAP"] = net_accuracy
        net_accuracy = self.test_net()
        client_param_dict["mAP"] = net_accuracy.item()
        # print(net_accuracy)
        log_info = '\t Round-{:>2d} | \t Cli-{:>2d} \t | \t ceLoss:{:.6f} | \t acc:{:.2f}%'.format(
            round + 1, self.client_id, loss, net_accuracy * 100
        )
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
                # print(param)
                # print(mask == 1)
                param.data[mask == 1] = self.residual[name].data[mask == 1]
                # print(param)

    # def test_net(self):
    #     if self.algorithm == "FedPE":
    #         return self.test_net_as_map()

    def test_net(self):
        self.net.to(self.device)
        self.net.eval()
        correct = 0
        test_data = DataLoader(self.test_dataset, batch_size=self.batch_size,
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
        accuracy = correct / len(test_data.dataset)
        self.net.to("cpu")
        return accuracy

    def test_net_as_map(self):
        self.net.to(self.device)
        self.net.eval()
        test_data = DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=False)
        # print(f"batch_size: {self.batch_size}")
        # 用于存储每个类别的统计
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for _, (data, labels) in enumerate(test_data):
                data, labels = data.to(self.device), labels.to(self.device)
                # print(f"{data.shape}")
                outputs = self.net(data)
                # print(f"data: {data}, shape: {data.shape} ------ label: {labels}, shape: {labels.shape}")
                # print(f"outputs: {outputs}, shape: {outputs.shape}")
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = outputs.data.max(1, keepdim=True)[1]

                # 遍历当前批次中的样本
                for label, pred in zip(labels, preds):
                    label = label.item()
                    # print(f"label:{label} ----- pred:{pred.item()}")
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1  # 累计该类别的样本数
                    if label == pred.item():
                        class_correct[label] += 1  # 累计正确预测的样本数

        # 计算加权类别准确率
        total_classes = len(class_total)  # 类别总数
        # print("Cli-{} | class_type:{}".format(self.client_id, class_total.keys()))
        average_precision = sum(class_correct[c] / class_total[c] for c in class_total.keys())
        mean_average_precision = average_precision / total_classes

        return mean_average_precision


if __name__ == "__main__":
    print(torch.sigmoid(torch.tensor(18 * 0.025)))

