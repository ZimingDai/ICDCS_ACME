#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   data.py
@Time    :   2024/06/14 16:04:07
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   允许对CIFAR10、CIFAR100、Stanford_Car进行预处理和加载
"""

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, StanfordCars
import os
import scipy.io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_dataset(name):
    if name == 'CIFAR10':
        return CIFAR10, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif name == 'CIFAR100':
        return CIFAR100, [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif name == 'stanford_car':
        return StanfordCars, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError("Unsupported dataset. Choose 'CIFAR10', 'CIFAR100', or 'StanfordCars'.")


def get_loaders(args):
    """
    获取数据加载器
    :param args: 参数对象，包含数据路径、批量大小和数据集名称等参数
    :return: 训练数据加载器、奖励数据加载器、验证数据加载器
    """
    dataset_class, mean, std = get_dataset(args.dataset_name)

    # 定义训练数据的预处理方式
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset_name == 'stanford_car':
        # 加载训练数据集，应用预处理方式
        train_dataset = dataset_class(
            root=args.data,
            split='train',
            download=False,
            transform=train_transform
        )
    else:
        # 加载训练数据集，应用预处理方式
        train_dataset = dataset_class(
            root=args.data,
            train=True,
            download=False,
            transform=train_transform,
        )

    indices = list(range(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:-5000]),
        pin_memory=True,
        num_workers=2,
    )

    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[-5000:]),
        pin_memory=True,
        num_workers=2,
    )

    # 定义验证数据的预处理方式
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset_name == 'stanford_car':
        # 加载训练数据集，应用预处理方式
        valid_dataset = dataset_class(
            root=args.data,
            split='test',
            download=False,
            transform=valid_transform
        )
    else:
        valid_dataset = dataset_class(
            root=args.data,
            train=False,
            download=False,
            transform=valid_transform,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    # 创建重复的数据加载器
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader


class RepeatedDataLoader():
    """
    重复的数据加载器类
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)
