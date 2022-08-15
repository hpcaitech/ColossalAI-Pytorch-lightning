import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from typing import Optional, Callable, Tuple, Any


class Cifar10Dataset(CIFAR10):
    def __init__(self, root: str, train: bool = True, pre_process: Optional[Callable] = None, transform=None, download: bool = False) -> None:
        super().__init__(root, train, transform, None, download)
        self.data = torch.tensor(self.data.transpose((0, 3, 1, 2)))
        self.targets = torch.tensor(self.targets)
        if pre_process:
            self.data = pre_process(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = img.cuda()
        if self.transform is not None:
            img = self.transform(img)
        return img, target.cuda()


def to_tensor(t):
    return t.float().div(255)


def build_data(batch_size: int):
    root = os.environ['DATA']
    transform_train = transforms.Compose([
        to_tensor,
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        to_tensor,
        transforms.Resize(32),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = Cifar10Dataset(root, train=True, transform=transform_train)
    valid_set = Cifar10Dataset(root, train=False, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader
