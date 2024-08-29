#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   motiv.py
@Time    :   2024/06/14 14:48:21
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   在16种不同的Backbone上运行不同Header
"""

from cloud_func import *
import argparse
import json
from vit_header_model import *

import setproctitle
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from typing import Optional, Union
from transformers.modeling_outputs import ImageClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 允许性能没有提升的最大 epoch 数
            min_delta (float): 性能提升的最小变化
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state_dict = None

    def __call__(self, val_accuracy, model):
        """
        Args:
            val_accuracy (float): 当前 epoch 的验证准确性
            model (nn.Module): 当前模型
        """
        if self.best_score is None:
            self.best_score = val_accuracy
            self.best_model_state_dict = model.state_dict()
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.best_model_state_dict = model.state_dict()
            self.counter = 0


def freeze_vit_backbone(model):
    for param in model.vit.parameters():
        param.requires_grad = False


# 定义一个函数，用于同时打印到终端和文件
def print_to_terminal_and_file(message, file):
    print(message)  # 打印到终端
    print(message, file=file)  # 写入文件


def main():
    # 命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--head_name", default=None, type=str, required=True, help="The type of the header.")

    parser.add_argument("--device", default=0, type=str, help="Choose the device to train.")

    parser.add_argument("--epoch", default=10, type=int, help="Epoch")

    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")

    parser.add_argument("--seed", default=0, type=int, help="Random Seed")
    parser.add_argument('--depth_mult_list', type=str, default='1.',
                        help="The possible depths used for training, e.g., '1.' is for default")

    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="The possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")

    parser.add_argument("--model_dir", default="../model/vit-base-patch16-224", type=str,
                        help="The pretrained model directory.")

    parser.add_argument("--freeze_backbone", action='store_true',
                        help="When fine-tuning, whether to freeze backbone")
    parser.add_argument("--data_name", type=str, default='stanford_car',
                        help="")

    args = parser.parse_args()

    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]
    args.task_name = "motiv"
    args.output_dir = "../moti/dynavit/"
    args.n_gpu = 1
    args.per_gpu_eval_batch_size = args.batch_size
    args.eval_batch_size = args.batch_size

    set_seed(args)

    # 打开Tensorboard

    writer = SummaryWriter(f"../runs/{args.head_name}_stanford_motiv")

    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{args.device}")
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available, using CPU instead.")
        current_device = 'cpu'
    # 定义使用的设备
    device = torch.device(current_device)
    args.device = device

    num_labels = 196

    with open(f'../log/Compare_Header_part/stanford_{args.head_name}.txt', 'a') as file_dir:

        for depth_mult in sorted(args.depth_mult_list, reverse=True):
            for width_mult in sorted(args.width_mult_list, reverse=True):
                print(f"width and depth: {width_mult}, {depth_mult}")
                # 准备模型、特征提取器和配置
                config = ViTConfig.from_pretrained(args.model_dir, num_labels=num_labels, output_hidden_states=True)
                feature_extractor = ViTImageProcessor.from_pretrained(args.model_dir)
                if args.head_name == 'linear':
                    model = ViTWithLinear.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'cnn':
                    model = ViTWithSimpleCNN.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'cnnproject':
                    model = ViTWithCNNProject.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'mixer':
                    model = ViTWithMixerBlock.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'cnnadd':
                    model = ViTWithCNNAdd.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'resmlp':
                    model = ViTWithResmlp.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'mlp':
                    model = ViTWithLayerNorm.from_pretrained(args.model_dir, num_classes=num_labels, config=config)
                elif args.head_name == 'vit':
                    model = ViTWithTransformerBlock.from_pretrained(args.model_dir, num_classes=num_labels,
                                                                    config=config)

                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))

                if args.freeze_backbone:
                    freeze_vit_backbone(model)

                train_dataset = load_and_cache_examples(feature_extractor, args.data_name, evaluate=False)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

                model.to(device)

                # 设置优化器，这里使用Adam优化器
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

                best_accuracy = 0.0  # 记录最佳准确率
                best_model_state_dict = None  # 用于保存最佳模型的状态字典  

                early_stopping = EarlyStopping(patience=3, min_delta=0.01)

                for epoch in range(args.epoch):
                    total_correct = 0
                    total = 0

                    for i, batch in enumerate(train_loader):  # 从数据加载器中迭代取出数据
                        images, labels = batch  # 获取图像和标签
                        images, labels = images.to(device), labels.to(device)  # 将数据移至设备

                        # 前向传播：计算模型输出
                        outputs = model(images, labels=labels)
                        logits = outputs.logits
                        loss = outputs.loss  # 计算损失

                        # 计算主分类器的准确性
                        _, predicted = torch.max(logits, 1)
                        total_correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        # 反向传播和优化
                        optimizer.zero_grad()  # 清空过去的梯度
                        loss.backward()  # 计算损失的梯度
                        optimizer.step()  # 根据梯度更新模型参数

                        print(f"Epoch {epoch}, Loss: {loss.item()}")  # 打印训练损失
                        # 将损失记录到TensorBoard
                        writer.add_scalar(f'Loss/w,d:{width_mult}, {depth_mult}', loss.item(),
                                          epoch * len(train_loader) + i)

                    # 在epoch结束时记录平均准确性
                    main_accuracy = total_correct / total
                    writer.add_scalar(f'Accuracy/w,d:{width_mult}, {depth_mult}', main_accuracy, epoch)
                    # 检查是否是最佳模型，如果是，则保存模型权重
                    if main_accuracy > best_accuracy:
                        best_accuracy = main_accuracy
                        best_model_state_dict = model.state_dict()  # 保存最佳模型的状态字典
                        print(f"Best model updated with accuracy: {best_accuracy}")
                    # 调用早停对象，检查是否满足早停条件
                    early_stopping(main_accuracy, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                # 加载最佳模型进行评估
                model.load_state_dict(best_model_state_dict)

                results = evaluate(args, model, feature_extractor)
                print_to_terminal_and_file("********** start evaluate results *********", file=file_dir)
                print_to_terminal_and_file(f"depth_mult: {depth_mult}", file=file_dir)
                print_to_terminal_and_file(f"width_mult: {width_mult}", file=file_dir)
                print_to_terminal_and_file(f"results: {results}", file=file_dir)
                print_to_terminal_and_file("********** end evaluate results *********", file=file_dir)


if __name__ == "__main__":
    main()
