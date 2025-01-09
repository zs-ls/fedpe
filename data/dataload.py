import random

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class DataSet:
    def __init__(self, name, distribution):
        self.name = name
        self.distribution = distribution
        self.train_set = None
        self.test_set = None
        self.all_clients_train_data = []
        self.all_clients_test_data = []

    def load(self):
        if self.name == "MINIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_set = torchvision.datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
            self.test_set = torchvision.datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
        return self

    def allocate(self, client_nums, train_samples_per_client, test_samples_per_client):
        if self.distribution == "IID":
            self.load() \
                .random_iid_data_split(client_nums, train_samples_per_client, test_samples_per_client)
            return self.all_clients_train_data, self.all_clients_test_data

    def random_iid_data_split(self, client_nums, train_samples_per_client, test_samples_per_client):
        train_indices = list(range(len(self.train_set)))
        test_indices = list(range(len(self.test_set)))

        random.shuffle(train_indices)
        random.shuffle(test_indices)

        for i in range(client_nums):
            train_start_idx = i * train_samples_per_client
            train_end_idx = train_start_idx + train_samples_per_client
            test_start_idx = i * test_samples_per_client
            test_end_idx = test_start_idx + test_samples_per_client
            train_data_set = torch.utils.data.Subset(self.train_set, train_indices[train_start_idx:train_end_idx])
            test_data_set = torch.utils.data.Subset(self.test_set, test_indices[test_start_idx:test_end_idx])
            self.all_clients_train_data.append(train_data_set)
            self.all_clients_test_data.append(test_data_set)


if __name__ == "__main__":
    client_num = 10
    train_samples_per_client = 200
    test_samples_per_client = 50
    all_clients_train_set, all_client_test_set = \
        DataSet("MINIST", "IID").allocate(client_num, train_samples_per_client, test_samples_per_client)

    # 打印数据集长度（数据集包含多少样本）
    for i in range(client_num):
        train_set = all_clients_train_set[i]
        test_set = all_client_test_set[i]
        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")

        # 获取一个样本图像和标签
        image, label = train_set[0]  # 获取第一个样本（图像，标签）
        print(image)

        # 打印图像形状
        print(f"Image shape: {image.shape}")  # 例如：torch.Size([1, 28, 28])

        # 可视化图像
        plt.imshow(image[0], cmap='gray')  # 显示图像 (MNIST 是灰度图，image[0] 选择第一个通道)
        plt.title(f"Label: {label}")
        plt.show()