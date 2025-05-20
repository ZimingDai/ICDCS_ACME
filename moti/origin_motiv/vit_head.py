#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   vit_head.py
@Time    :   2024/06/14 14:48:21
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   对ViT的宽度进行修改的测试代码
"""
import argparse
import random
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def reorder_neuoron_and_head(model, head_importance, neuron_importance):
    base_model = getattr(model, model.base_model_prefix, model)

    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)


def main(file_dir):
    # 命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default=1, type=str, help="Choose the device to train.")

    parser.add_argument("--imp_epoch", default=1, type=int,
                        help="the model's train epoch to calculate the importance of heads and neurons, 1 is default.")

    parser.add_argument("--epoch", default=5, type=int,
                        help="the model's train epoch to improve performance, 5 is default.")

    parser.add_argument("--batch_size", default=32, type=int, help="batch size")

    parser.add_argument("--reorder", action="store_true", help="whether to reorder heads and neuron")

    parser.add_argument("--seed", default=42, type=int, help="random seed")

    parser.add_argument('--width_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")

    args = parser.parse_args()

    set_seed(args)

    print(
        f"Reorder: {args.reorder}, Width: {args.width_mult}, Imp_epoch: {args.imp_epoch}, Epoch: {args.epoch}, Batch_size: {args.batch_size}, Seed: {args.seed}",
        file=file_dir)

    # 检查并打印GPU信息
    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{args.device}")
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available, using CPU instead.")
        current_device = 'cpu'

    # 定义使用的设备
    device = torch.device(current_device)

    # 数据预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224以符合ViT模型输入要求
        transforms.ToTensor(),  # 将图像数据转换为PyTorch张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图像数据标准化
    ])

    # 加载CIFAR-100训练集
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 创建数据加载器用于批量处理和打乱数据

    # 加载CIFAR-100测试集
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # 创建测试数据加载器

    model = ViTForImageClassification.from_pretrained('./vit-base-patch16-224')

    # 替换模型的分类层以匹配CIFAR-100的类别数目（100个类别）
    model.classifier = nn.Linear(model.classifier.in_features, 100)
    # 将模型移至定义的设备上（GPU或CPU）
    model.to(device)

    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads, device=device)
    head_mask.requires_grad_(True)

    # 设置优化器，这里使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # 设置损失函数，这里使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # collect weights
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(device))

    # 计算模型heads与neuron重要性
    model.train()  # 将模型设置为训练模式
    for epoch in tqdm(range(args.imp_epoch), desc="The computing epochs"):  # 迭代训练周期
        with tqdm(train_loader, desc="Batches", leave=False) as tbatch:
            for batch in tbatch:  # 从数据加载器中迭代取出数据
                images, labels = batch  # 获取图像和标签
                images, labels = images.to(device), labels.to(device)  # 将数据移至设备

                # 前向传播：计算模型输出
                outputs = model(images, head_mask=head_mask).logits

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()  # 清空过去的梯度
                loss.backward()  # 计算损失的梯度
                head_importance += head_mask.grad.abs().detach()  # 获取头部掩码的梯度的绝对值

                # calculate  neuron importance
                for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight,
                                                          neuron_importance):
                    current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
                    current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

                optimizer.step()  # 根据梯度更新模型参数
                tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")

    if args.reorder == True:
        # 按照模型中神经元的重要程度进行重构
        reorder_neuoron_and_head(model, head_importance, neuron_importance)

    # print("Heads' gradients: ", head_importance)
    # print("neuron's importance:", neuron_importance)

    model.apply(lambda m: setattr(m, 'width_mult', float(args.width_mult)))  # 设置宽度

    model.train()  # 将模型设置为训练模式
    for epoch in tqdm(range(args.epoch), desc="The traing epochs"):  # 迭代训练周期
        with tqdm(train_loader, desc="Batches", leave=False) as tbatch:
            for batch in tbatch:  # 从数据加载器中迭代取出数据
                images, labels = batch  # 获取图像和标签
                images, labels = images.to(device), labels.to(device)  # 将数据移至设备

                # 前向传播：计算模型输出
                outputs = model(images).logits

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()  # 清空过去的梯度
                loss.backward()  # 计算损失的梯度

                optimizer.step()  # 根据梯度更新模型参数
                tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")

    # 评估模型性能
    model.eval()  # 将模型设置为评估模式
    total = 0  # 记录总样本数
    correct = 0  # 记录正确预测的样本数
    with torch.no_grad():  # 测试阶段不计算梯度
        for batch in test_loader:  # 从数据加载器中迭代取出数据
            images, labels = batch  # 获取图像和标签
            images, labels = images.to(device), labels.to(device)  # 将数据移至设备

            outputs = model(images).logits  # 前向传播：计算模型输出
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    print(f"Accuracy on CIFAR-100 test set: {100 * correct / total}%", file=file_dir)


if __name__ == "__main__":
    with open('log/output_head.txt', 'a') as file:
        main(file)
