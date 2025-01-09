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

    def train(self, global_net):
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
            lr=0.01,  # 只设置学习率
        )

        data_loader = DataLoader(self.train_dataset, 5, shuffle=True, drop_last=True)
        loss_history = []
        for e in range(self.epoch):
            losses = []
            for b_idx, (images, labels) in enumerate(data_loader):
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
