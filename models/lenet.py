import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, class_num):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = LeNet(10)
    model_dict = model.state_dict()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(name)
    for name, param in model_dict.items():
        if "weight" in name:
            print(name)