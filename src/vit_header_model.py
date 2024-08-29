#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cloud_func.py
@Time    :   2024/06/26 09:33:17
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   所有ViT+各种Header的模型定义
"""
from cloud_func import *

import setproctitle
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from typing import Optional, Union
from transformers.modeling_outputs import ImageClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class ViTWithLinear(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量
        self.classifier = nn.Linear(config.hidden_size, num_classes)

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

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        print("Linear logits:", logits)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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
        assert patch_size * patch_size == (
                    seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

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


class CNNProject(nn.Module):
    def __init__(self, channels, image_size, patch_size, num_classes):
        super(CNNProject, self).__init__()
        self.width = self.height = image_size // patch_size
        self.channels = channels

        self.conv = nn.Conv2d(self.channels * 2, 16, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (self.width // 2) * (self.height // 2), num_classes)

    def forward(self, x, cls_token):
        # Concatenate the CLS token with every patch
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1)  # Reshape CLS token to match the spatial dimensions
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, self.width, self.height)

        x = torch.cat([cls_token.expand(-1, -1, self.width, self.height), x], dim=1)
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ViTWithCNNProject(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量

        self.classifier = CNNProject(
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

        hidden_states = outputs.hidden_states  # 获取所有隐藏层的状态
        cls_token = hidden_states[-1][:, 0, :]  # 取出 [CLS] token

        # 取出每个样本的最后一个隐藏状态，去掉[CLS] token
        hidden_state = hidden_states[-1]
        batch_size, seq_len, hidden_size = hidden_state.size()
        patch_size = int((seq_len - 1) ** 0.5)
        assert patch_size * patch_size == (
                    seq_len - 1), f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

        cnn_input = hidden_state[:, 1:, :].view(batch_size, hidden_size, patch_size, patch_size)

        logits = self.classifier(cnn_input, cls_token)

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


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, activation=nn.GELU(), **kwargs):
        super(MlpBlock, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, channel_dim, token_mixer_hidden_dim, channel_mixer_hidden_dim=None,
                 activation=nn.GELU(), **kwargs):
        super(MixerBlock, self).__init__()

        if channel_mixer_hidden_dim is None:
            channel_mixer_hidden_dim = token_mixer_hidden_dim

        self.num_patches = num_patches
        self.channel_dim = channel_dim
        self.token_mixer_hidden_dim = token_mixer_hidden_dim
        self.channel_mixer_hidden_dim = channel_mixer_hidden_dim
        self.activation = activation

        self.norm1 = nn.LayerNorm(channel_dim)
        self.token_mixer = MlpBlock(num_patches, token_mixer_hidden_dim, activation)

        self.norm2 = nn.LayerNorm(channel_dim)
        self.channel_mixer = MlpBlock(channel_dim, channel_mixer_hidden_dim, activation)

    def forward(self, x):
        skip_x = x  # 保存跳跃连接的输入
        x = self.norm1(x)  # 在x的最后一个维度进行归一化
        x = x.permute(0, 2, 1)  # 调整x的形状以匹配token_mixer的输入要求 (batch_size, channel_dim, num_patches)
        x = self.token_mixer(x)  # 应用token_mixer
        x = x.permute(0, 2, 1)  # 恢复x的形状 (batch_size, num_patches, channel_dim)
        x = x + skip_x  # 添加跳跃连接
        skip_x = x  # 更新跳跃连接输入

        x = self.norm2(x)  # 在x的最后一个维度进行归一化
        x = self.channel_mixer(x)  # 应用channel_mixer
        x = x + skip_x  # 添加跳跃连接

        return x


class ViTWithMixerBlock(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量

        self.classifier = MixerBlock(
            num_patches=(config.image_size // config.patch_size) ** 2 + 1,
            channel_dim=config.hidden_size,
            token_mixer_hidden_dim=config.hidden_size * 4,  # 隐藏层大小可以根据需要调整
            activation=nn.GELU()
        )

        # 添加一个全连接层将输出维度从hidden_size转换为num_classes
        self.fc = nn.Linear(config.hidden_size, num_classes)

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

        # 取出每个样本的最后一个隐藏状态，包括[CLS] token
        batch_size, seq_len, hidden_size = hidden_state.size()
        patch_size = int((seq_len - 1) ** 0.5)
        assert patch_size * patch_size + 1 == seq_len, f"Patch size {patch_size} does not match sequence length {seq_len}"

        mixer_input = hidden_state.view(batch_size, seq_len, hidden_size)

        # 使用自定义的MixerBlock进行分类
        x = self.classifier(mixer_input)

        # 取CLS token的输出并应用全连接层
        logits = self.fc(x[:, 0, :])

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


class CNNAdd(nn.Module):
    def __init__(self, channels, image_size, patch_size, num_classes):
        super(CNNAdd, self).__init__()
        self.width = self.height = image_size // patch_size
        self.channels = channels

        self.conv = nn.Conv2d(self.channels, 16, kernel_size=3, padding=1, bias=False)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (self.width // 2) * (self.height // 2), num_classes)

    def forward(self, x):
        # Remove the CLS token and reshape to image dimensions
        y1 = x[:, 1:, :].view(-1, self.channels, self.width, self.height)

        # Extract the CLS token, repeat it, and reshape to image dimensions
        y2 = x[:, 0, :].unsqueeze(1).repeat(1, self.width * self.height, 1)
        y2 = y2.view(-1, self.channels, self.width, self.height)

        # Add the two components
        y = y1 + y2

        # Apply the convolutional layers
        y = self.conv(y)
        y = self.elu(y)
        y = self.pool(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y


class ViTWithCNNAdd(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)
        self.num_classes = num_classes

        self.classifier = CNNAdd(
            channels=config.hidden_size,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=num_classes
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

        # Ensure the patch size and sequence length match
        batch_size, seq_len, hidden_size = hidden_state.size()
        patch_size = int((seq_len - 1) ** 0.5)
        assert patch_size * patch_size + 1 == seq_len, f"Patch size {patch_size} does not match sequence length {seq_len}"

        # Pass through the CNNAdd classifier
        logits = self.classifier(hidden_state)

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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


class ViTWithResmlp(ViTForImageClassification):
    def __init__(self, config, num_classes=100, mlp_dim=3072):
        super().__init__(config)
        self.mlp_mixer = MlpBlock(dim=config.hidden_size, hidden_dim=mlp_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config.hidden_size, num_classes)

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
        assert patch_size * patch_size + 1 == seq_len, f"Patch size {patch_size} does not match sequence length {seq_len - 1}"

        # Pass through the MlpBlock
        y = self.mlp_mixer(hidden_state[:, 1:, :])

        # Apply global average pooling
        y = y.permute(0, 2, 1)  # 交换维度以适应 pooling 的要求
        y = self.global_avg_pool(y).squeeze(-1)

        # Fully connected layer for classification
        logits = self.fc(y)

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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


class ViTWithLayerNorm(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.fc = nn.Linear(config.hidden_size, num_classes)

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

        # Apply LayerNorm
        y = self.layer_norm(hidden_state)

        # Extract [CLS] token
        cls_token = y[:, 0]

        # Fully connected layer for classification
        logits = self.fc(cls_token)

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(mlp_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x


class ViTWithTransformerBlock(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)
        self.transformer_block = TransformerBlock(num_heads=12, mlp_dim=config.hidden_size, dropout=0.1)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.fc = nn.Linear(config.hidden_size, num_classes)

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

        # Apply TransformerBlock
        y = self.transformer_block(hidden_state)

        # Apply LayerNorm
        y = self.layer_norm(y)

        # Extract [CLS] token
        cls_token = y[:, 0]

        # Fully connected layer for classification
        logits = self.fc(cls_token)

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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
