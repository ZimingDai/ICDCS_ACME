#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   micro_child.py
@Time    :   2024/06/14 15:37:51
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   定义NAS生成的Header框架以及ViT对接框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTModel, ViTConfig
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class ENASCNN(nn.Module):
    def __init__(self, args, hidden_size, num_class):
        super(ENASCNN, self).__init__()
        self.args = args
        self.num_layers = args.child_num_layers
        self.out_filters = args.child_out_filters
        self.num_branches = args.child_num_branches
        self.num_cells = args.child_num_cells
        self.use_aux_heads = args.child_use_aux_heads
        self.hidden_size = hidden_size
        self.fixed_arc = None
        self.num_class = num_class

        if self.use_aux_heads:
            self.aux_head_indices = [self.num_layers // 2]

        self.stem_conv = nn.Sequential(
            nn.Conv2d(hidden_size, self.out_filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_filters),
        )
        self._compile_model()
        self._init_param(self.modules())

    def forward(self, inputs, dag):
        self.normal_arc = dag
        logi, aux_logits = self._get_model(inputs)
        return logi, aux_logits

    def _compile_model(self):
        out_filters = self.out_filters
        in_filters = [self.hidden_size, self.out_filters]
        self.add_module('layer', nn.ModuleList())
        for layer_id in range(self.num_layers + 2):
            self.layer.append(nn.Module())
            self._compile_layer(self.layer[layer_id], layer_id, in_filters, out_filters)
            in_filters = [in_filters[-1], out_filters]

        self.add_module('final_fc', nn.Linear(out_filters, self.num_class))

    def _compile_layer(self, module, layer_id, in_filters, out_filters):  # 编译层函数
        self._compile_calibrate(module, in_filters, out_filters)  # 编译校准模块
        module.add_module('cell', nn.ModuleList())  # 添加ModuleList用于存储单元
        for cell_id in range(self.num_cells):  # 遍历所有单元
            module.cell.append(nn.ModuleList())  # 添加一个空的模块列表
            for i in range(2):  # 遍历两个分支
                module.cell[cell_id].append(nn.Module())  # 添加一个空的模块
                self._compile_cell(module.cell[cell_id][i], cell_id, out_filters)  # 编译单元

        # 直接通过标准正态分布随机数进行初始化
        param = nn.Parameter(torch.randn(self.num_cells + 2, out_filters ** 2, 1, 1) * 0.01)
        module.register_parameter('final_conv', param)  # 注册最终的卷积参数
        module.add_module('final_bn', nn.BatchNorm2d(out_filters, track_running_stats=False))  # 添加最终的批量归一化层

    def _compile_cell(self, module, curr_cell, out_filters):
        module.add_module('three', nn.ModuleList())
        self._compile_conv(module.three, curr_cell, 3, out_filters)
        module.add_module('five', nn.ModuleList())
        self._compile_conv(module.five, curr_cell, 5, out_filters)
        module.add_module('one', nn.ModuleList())
        self._compile_conv(module.one, curr_cell, 1, out_filters)
        # module.add_module('layernorm', nn.ModuleList())
        # self._compile_layernorm(module.layernorm, curr_cell, out_filters)  

    def _compile_conv(self, module, curr_cell, filter_size, out_filters):
        num_possible_inputs = curr_cell + 2
        for i in range(num_possible_inputs):
            module.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(out_filters, out_filters, filter_size, padding=filter_size // 2, groups=out_filters,
                          bias=False),
                nn.Conv2d(out_filters, out_filters, 1, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(out_filters, out_filters, filter_size, padding=filter_size // 2, groups=out_filters,
                          bias=False),
                nn.Conv2d(out_filters, out_filters, 1, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))

    # def _compile_layernorm(self, module, curr_cell, out_filters):
    #     num_possible_inputs = curr_cell + 2  # 确保有足够的 LayerNorm 模块
    #     for i in range(num_possible_inputs):
    #         module.append(nn.LayerNorm([out_filters, 32, 32]))  # 假设输入的形状为 [out_filters, 32, 32]

    def _compile_calibrate(self, module, in_filters, out_filters):
        module.add_module('calibrate', nn.Module())
        if in_filters[0] != out_filters:
            module.calibrate.add_module('pool_x', nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_filters[0], out_filters, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))
        if in_filters[1] != out_filters:
            module.calibrate.add_module('pool_y', nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_filters[1], out_filters, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))

    def _get_model(self, inputs):
        aux_logits = None
        x = self.stem_conv(inputs)
        layers = [x, x]

        out_filters = self.out_filters
        for layer_id in range(self.num_layers + 2):
            if self.fixed_arc is None:
                x = self._enas_layer(layers, self.layer[layer_id], self.normal_arc, out_filters)
            layers = [layers[-1], x]

        x = F.dropout2d(F.adaptive_avg_pool2d(F.relu(x), 1), 0.1)
        x = self.final_fc(x.view(x.size(0), -1))
        return x, aux_logits

    def _enas_layer(self, prev_layers, module, arc, out_filters):
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, module.calibrate, out_filters)
        used = []
        for cell_id in range(self.num_cells):
            prev_layers = torch.stack(layers)
            x_id = arc[4 * cell_id]  # 获取x的ID
            x_op = arc[4 * cell_id + 1]  # 获取x的操作
            x = prev_layers[x_id, :, :, :, :]
            x = self._enas_cell(x, module.cell[cell_id][0], cell_id, x_id, x_op)
            x_used = torch.zeros(self.num_cells + 2).long()
            x_used[x_id] = 1

            y_id = arc[4 * cell_id + 2]  # 获取y的ID
            y_op = arc[4 * cell_id + 3]  # 获取y的操作
            y = prev_layers[y_id, :, :, :, :]
            y = self._enas_cell(y, module.cell[cell_id][1], cell_id, y_id, y_op)
            y_used = torch.zeros(self.num_cells + 2).long()
            y_used[y_id] = 1
            out = x + y
            used.extend([x_used, y_used])
            layers.append(out)

        used_ = torch.zeros(used[0].shape).long()  # 初始化使用的列表
        for i in range(len(used)):
            used_ = used_ + used[i]
        indices = torch.eq(used_, 0).nonzero().long().view(-1)
        num_outs = indices.size(0)
        out = torch.stack(layers)
        out = out[indices]

        inp = prev_layers[0]
        N, C, H, W = inp.shape
        out = out.transpose(0, 1).contiguous().view(N, num_outs * out_filters, H, W)

        out = F.relu(out)
        w = module.final_conv[indices].view(out_filters, out_filters * num_outs, 1, 1)  # TODO：w 全是0？？？
        out = F.conv2d(out, w)
        out = module.final_bn(out)

        out = out.view(prev_layers[0].shape)
        return out

    def _maybe_calibrate_size(self, layers, module, out_filters):  # 校准层大小函数
        hw = [layer.shape[2] for layer in layers]  # 获取层的高度和宽度
        c = [layer.shape[1] for layer in layers]  # 获取层的通道数

        x = layers[0]  # 获取第一个层的输出
        if c[0] != out_filters:  # 如果通道数不等于输出过滤器数
            x = module.pool_x(x)  # 应用池化
        y = layers[1]  # 获取第二个层的输出
        if c[1] != out_filters:  # 如果通道数不等于输出过滤器数
            y = module.pool_y(y)  # 应用池化
        return [x, y]  # 返回校准后的层

    def _enas_cell(self, x, module, curr_cell, prev_cell, op_id):
        if op_id == 0:
            out = module.three[prev_cell](x)
        elif op_id == 1:
            out = module.five[prev_cell](x)
        elif op_id == 2:
            out = F.avg_pool2d(x, 3, stride=1, padding=1)
        elif op_id == 3:
            out = F.max_pool2d(x, 3, stride=1, padding=1)
        elif op_id == 4:
            out = module.one[prev_cell](x)

        else:
            out = x
        return out

    def reset_parameters(self):
        pass

    def _init_param(self, module, trainable=True, seed=None):
        for mod in module:
            if type(mod) == nn.Conv2d or type(mod) == nn.Linear:
                nn.init.kaiming_normal_(mod.weight)


class ViTWithENASCNN(ViTForImageClassification):
    def __init__(self, config, args, num_classes):
        super().__init__(config)
        self.num_classes = num_classes
        self.classifier = ENASCNN(args, hidden_size=config.hidden_size, num_class=num_classes)

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

        logits, _ = self.classifier(cnn_input, dag)

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
