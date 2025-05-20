#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cloud_pretrain.py
@Time    :   2024/06/14 15:18:38
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   对DynaViTw进行蒸馏的最初等的ViT预训练（First Step）
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor
from torch.optim import AdamW
import os
from tqdm import tqdm

NAME = 'stanford_car'  # 'cifar100' or 'cifar10'

if NAME == 'stanford_car':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.StanfordCars(root='../data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_label = 196
elif NAME == 'cifar10':
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    # 加载 CIFAR10 数据集
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_label = 10

elif NAME == 'cifar100':
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    # 加载 CIFAR100 数据集
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_label = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Vision Transformer 模型
config = ViTConfig.from_pretrained('../model/vit-base-patch16-224', num_labels=num_label)
model = ViTForImageClassification.from_pretrained('../model/vit-base-patch16-224', config=config)
model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 特征提取器
feature_extractor = ViTFeatureExtractor.from_pretrained('../model/vit-base-patch16-224')

# 训练模型
model.train()
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存模型和特征提取器配置
model_path = f'../model/vit_{NAME}_model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_pretrained(model_path, safe_serialization=False)
feature_extractor.save_pretrained(model_path)

print("Model and preprocessor configuration saved to", model_path)
