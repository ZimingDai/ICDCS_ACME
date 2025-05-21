#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   micro_child.py
@Time    :   2024/06/14 15:37:51
@Author  :   PhoenixDai 
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   定义Header框架以及ViT框架
'''

import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from layers.gate_layer import GateLayer

class TempCNN(nn.Module):
    def __init__(self, args, hidden_size):
        super(TempCNN, self).__init__()
        self.args = args
        self.out_filters = 256
        self.hidden_size = hidden_size
        self.five = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.out_filters, self.out_filters, 5, padding=5 // 2, groups=self.out_filters, bias=False),
                GateLayer(self.out_filters,self.out_filters,[1, -1, 1, 1]),
                nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
                GateLayer(self.out_filters, self.out_filters, [1, -1, 1, 1]),
                nn.BatchNorm2d(self.out_filters, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(self.out_filters, self.out_filters, 5, padding=5 // 2, groups=self.out_filters, bias=False),
                GateLayer(self.out_filters, self.out_filters, [1, -1, 1, 1]),
                nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
                GateLayer(self.out_filters, self.out_filters, [1, -1, 1, 1]),
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

        x = self.max_pool(i)
        y = self.five(i)
        i = x + y

        x = self.max_pool(i)
        y = self.five(i)
        z = x + y

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
        assert patch_size * patch_size == (seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

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

class TempCNN1(nn.Module):
    def __init__(self, args, hidden_size):
        super(TempCNN1, self).__init__()
        self.args = args
        self.out_filters = 256
        self.hidden_size = hidden_size
        self.five = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_filters, self.out_filters, 5, padding=5 // 2, groups=self.out_filters, bias=False),
            GateLayer(self.out_filters, self.out_filters, [1, -1, 1, 1]),
            nn.Conv2d(self.out_filters, self.out_filters, 1, bias=False),
            GateLayer(self.out_filters, self.out_filters, [1, -1, 1, 1]),
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
        y = self.five(i)
        z = y
        z = self.relu(z)
        z = self.adaptive_pool(z)
        z = self.dropout(z)
        z = self.final_fc(z.view(z.size(0), -1))
        return z

class ViTWithTempCNN1(ViTForImageClassification):
    def __init__(self, config, args, num_classes=100):
        super().__init__(config)
        self.num_classes = num_classes
        self.classifier = TempCNN1(args, hidden_size=config.hidden_size)

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
        assert patch_size * patch_size == (seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

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

class SimpleCNN(nn.Module):
    def __init__(self, channels, image_size, patch_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.width = self.height = image_size // patch_size
        self.channels = channels

        self.conv = nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (self.width // 2) * (self.height // 2), num_classes)

    def forward(self, x):
        # Reshape the input from (batch_size, hidden_size) to (batch_size, channels, width, height)
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, self.width, self.height)

        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ViTWithSimpleCNN(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量

        self.classifier = SimpleCNN(
            channels=config.hidden_size,
            num_classes=config.num_labels,
            image_size=config.image_size,
            patch_size=config.patch_size
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        hidden_state = outputs.hidden_states[-1]

        # 取出每个样本的最后一个隐藏状态，去掉[CLS] token
        batch_size, seq_len, hidden_size = hidden_state.size()
        patch_size = int((seq_len - 1) ** 0.5)
        assert patch_size * patch_size == (seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

        cnn_input = hidden_state[:, 1:, :].view(batch_size, hidden_size, patch_size, patch_size)

        # 使用自定义的CNN进行分类
        logits = self.classifier(cnn_input)

        loss = None
        if labels is not None:
            # 将标签移动到与logits相同的设备上
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