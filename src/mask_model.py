#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cloud_func.py
@Time    :   2024/06/26 09:33:17
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   在代码中用来对接Prune的废弃代码（最开始尝试与第三步进行拼接，但是fail了）
"""
import torch.nn.functional as F
from cloud_func import *
import argparse
import json

import setproctitle
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from typing import Optional, Union
from transformers.modeling_outputs import ImageClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class TempCNN(nn.Module):
    def __init__(self, args, hidden_size):
        super(TempCNN, self).__init__()
        self.args = args
        self.out_filters = 256
        self.hidden_size = hidden_size
        self.five = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_filters, self.out_filters, 5, padding=5 // 2, groups=self.out_filters, bias=False),
            nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
            nn.BatchNorm2d(self.out_filters, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.out_filters, self.out_filters, 5, padding=5 // 2, groups=self.out_filters, bias=False),
            nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
            nn.BatchNorm2d(self.out_filters, track_running_stats=False),
        )
        self.three = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_filters, self.out_filters, 3, padding=3 // 2, groups=self.out_filters, bias=False),
            nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
            nn.BatchNorm2d(self.out_filters, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.out_filters, self.out_filters, 3, padding=3 // 2, groups=self.out_filters, bias=False),
            nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
            nn.BatchNorm2d(self.out_filters, track_running_stats=False),
        )

        self.stem_conv = nn.Sequential(
            nn.Conv2d(hidden_size, self.out_filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_filters),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(0.1)
        self.final_fc = nn.Linear(self.out_filters, 100)

    def forward(self, i):
        # Reshape the input from (batch_size, hidden_size) to (batch_size, channels, width, height)
        i = self.stem_conv(i)

        a = self.max_pool(i)
        b = self.five(i)
        z1 = a + b

        a = self.avg_pool(z1)
        b = self.max_pool(i)
        z2 = a + b

        a = self.three(i)
        b = self.three(i)
        z3 = a + b

        z = z1 + z2 + z3
        z = self.relu(z)
        z = self.adaptive_pool(z)
        z = self.dropout(z)
        z = self.final_fc(z.view(z.size(0), -1))
        return z


class ViTWithTempCNN(ViTForImageClassification):
    def __init__(self, config, args, num_classes=100):
        super().__init__(config)
        self.num_classes = num_classes
        self.classifier = TempCNN(args, hidden_size=config.hidden_size)

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            dag=None,
    ) -> Union[tuple, ImageClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # 获取最后一层隐藏状态
        hidden_state = outputs.hidden_states[-1]
        batch_size, seq_len, hidden_size = hidden_state.size()
        patch_size = int((seq_len - 1) ** 0.5)

        # 确保patch_size计算正确
        assert patch_size * patch_size == (
                    seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

        # 去除CLS token，并重新排列为适应CNN的格式
        cnn_input = hidden_state[:, 1:, :].view(batch_size, patch_size, patch_size, hidden_size).permute(0, 3, 1, 2)

        logits = self.classifier(cnn_input)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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

    parser.add_argument("--device", default=0, type=str, help="Choose the device to train.")

    parser.add_argument("--epoch", default=100, type=int, help="Epoch")

    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")

    parser.add_argument("--seed", default=0, type=int, help="Random Seed")

    parser.add_argument("--model_dir", default="../model/dynavit", type=str,
                        help="The pretrained model directory.")

    parser.add_argument("--freeze_backbone", action='store_true',
                        help="When fine-tuning, whether to freeze backbone")

    args = parser.parse_args()

    args.task_name = "motiv"
    args.n_gpu = 1
    args.output_dir = "../moti/dynavit/"
    args.per_gpu_eval_batch_size = args.batch_size
    args.eval_batch_size = args.batch_size

    set_seed(args)

    # 打开Tensorboard

    writer = SummaryWriter(f"../runs/tempcnn_{args.seed}_motiv_2")

    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{args.device}")
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available, using CPU instead.")
        current_device = 'cpu'
    # 定义使用的设备
    device = torch.device(current_device)
    args.device = device

    num_labels = 100

    with open(f'../log/Compare_Header_part/output_tempcnn_2.txt', 'a') as file_dir:

        # 准备模型、特征提取器和配置
        config = ViTConfig.from_pretrained(args.model_dir, num_labels=num_labels, output_hidden_states=True)
        feature_extractor = ViTImageProcessor.from_pretrained(args.model_dir)
        model = ViTWithTempCNN.from_pretrained(args.model_dir, config=config, args=args, num_classes=100)

        model.apply(lambda m: setattr(m, 'depth_mult', 0.25))
        model.apply(lambda m: setattr(m, 'width_mult', 0.25))

        if args.freeze_backbone:
            freeze_vit_backbone(model)

        train_dataset = load_and_cache_examples(feature_extractor, evaluate=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        model.to(device)

        # 设置优化器，这里使用Adam优化器
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

        best_accuracy = 0.0  # 记录最佳准确率
        best_model_state_dict = None  # 用于保存最佳模型的状态字典  

        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        model.train()

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
                writer.add_scalar(f'Loss/w,d:{0.25}, {0.25}', loss.item(), epoch * len(train_loader) + i)

            # 在epoch结束时记录平均准确性
            main_accuracy = total_correct / total
            writer.add_scalar(f'Accuracy/w,d:{0.25}, {0.25}', main_accuracy, epoch)
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
        print_to_terminal_and_file(f"results: {results}", file=file_dir)
        print_to_terminal_and_file("********** end evaluate results *********", file=file_dir)


if __name__ == "__main__":
    main()
