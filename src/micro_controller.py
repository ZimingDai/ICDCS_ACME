#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   micro_controller.py
@Time    :   2024/06/14 13:36:33
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   定义用来ENAS中作决策的LSTM
"""
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能模块


class Controller(torch.nn.Module):  # 定义Controller类，继承自torch.nn.Module
    def __init__(self, args):  # 初始化函数
        torch.nn.Module.__init__(self)  # 调用父类的初始化函数
        self.args = args  # 保存传入的参数
        self.num_branches = args.child_num_branches  # 获取子网络的分支数
        self.num_cells = args.child_num_cells  # 获取子网络的单元数
        self.lstm_size = args.lstm_size  # 获取LSTM的大小
        self.lstm_num_layers = args.lstm_num_layers  # 获取LSTM的层数
        self.lstm_keep_prob = args.lstm_keep_prob  # 获取LSTM的保持概率
        self.temperature = args.temperature  # 获取温度参数
        self.tanh_constant = args.controller_tanh_constant  # 获取tanh常数
        self.op_tanh_reduce = args.controller_op_tanh_reduce  # 获取操作tanh减少系数

        self.encoder = nn.Embedding(self.num_branches + 1, self.lstm_size)  # 定义嵌入层

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)  # 定义LSTM单元
        self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)  # 定义线性层，用于softmax
        b_soft = torch.zeros(1, self.num_branches)  # 初始化softmax的偏置
        b_soft[:, 0:2] = 10  # 设置前两个分支的偏置为10
        self.b_soft = nn.Parameter(b_soft)  # 将softmax的偏置设置为可学习参数
        b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.num_branches - 2))  # 定义一个不学习的softmax偏置
        b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.num_branches])  # 重塑softmax偏置的形状
        self.b_soft_no_learn = torch.Tensor(b_soft_no_learn).requires_grad_(False).cuda()  # 将softmax偏置转换为张量，并移到GPU上

        # 定义注意力机制的参数
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)  # 定义线性层，用于注意力机制
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)  # 定义线性层，用于注意力机制
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)  # 定义线性层，用于注意力机制

        self.reset_param()  # 初始化参数

    def reset_param(self):  # 初始化参数的函数
        for name, param in self.named_parameters():  # 遍历所有参数
            if 'b_soft' not in name:  # 如果参数名不包含'b_soft'
                nn.init.uniform_(param, -0.1, 0.1)  # 使用均匀分布初始化参数

    # def forward(self):  # 前向传播函数
    #     arc_seq_1, entropy_1, log_prob_1, c, h = self.run_sampler(use_bias=True)  # 第一次运行采样器
    #     arc_seq_2, entropy_2, log_prob_2, _, _ = self.run_sampler(prev_c=c, prev_h=h)  # 第二次运行采样器
    #     sample_arc = (arc_seq_1, arc_seq_2)  # 合并采样结果
    #     sample_entropy = entropy_1 + entropy_2  # 合并熵
    #     sample_log_prob = log_prob_1 + log_prob_2  # 合并对数概率
    #     return sample_arc, sample_log_prob, sample_entropy  # 返回采样结果、对数概率和熵

    def forward(self):  # 前向传播函数
        arc_seq_1, entropy_1, log_prob_1, c, h = self.run_sampler(use_bias=True)  # 第一次运行采样器
        sample_arc = arc_seq_1  # 合并采样结果
        sample_entropy = entropy_1  # 合并熵
        sample_log_prob = log_prob_1  # 合并对数概率
        return sample_arc, sample_log_prob, sample_entropy  # 返回采样结果、对数概率和熵

    def run_sampler(self, prev_c=None, prev_h=None, use_bias=False):  # 运行采样器的函数
        if prev_c is None:  # 如果前一个LSTM单元的c状态为空
            prev_c = torch.zeros(1, self.lstm_size).cuda()  # 初始化前一个LSTM单元的c状态，并移到GPU上
            prev_h = torch.zeros(1, self.lstm_size).cuda()  # 初始化前一个LSTM单元的h状态，并移到GPU上

        inputs = self.encoder(torch.zeros(1).long().cuda())  # 初始化输入，将其通过嵌入层，并移到GPU上

        anchors = []  # 初始化anchors列表
        anchors_w_1 = []  # 初始化anchors_w_1列表
        arc_seq = []  # 初始化arc_seq列表

        for layer_id in range(2):  # 遍历前两个层
            embed = inputs  # 获取输入
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))  # 计算下一个LSTM单元的h和c状态
            prev_c, prev_h = next_c, next_h  # 更新前一个LSTM单元的c和h状态
            anchors.append(torch.zeros(next_h.shape).cuda())  # 将下一层的h状态初始化为0，并移到GPU上，添加到anchors列表中
            anchors_w_1.append(self.w_attn_1(next_h))  # 计算注意力机制的w_1，添加到anchors_w_1列表中

        layer_id = 2  # 初始化层ID为2
        entropy = []  # 初始化熵列表
        log_prob = []  # 初始化对数概率列表

        while layer_id < self.num_cells + 2:  # 遍历所有单元
            indices = torch.arange(0, layer_id).cuda()  # 获取当前层的索引，并移到GPU上
            start_id = 4 * (layer_id - 2)  # 计算开始ID
            prev_layers = []  # 初始化前一层的列表
            for i in range(2):  # 遍历两个索引
                embed = inputs  # 获取输入
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))  # 计算下一个LSTM单元的h和c状态
                prev_c, prev_h = next_c, next_h  # 更新前一个LSTM单元的c和h状态
                query = torch.stack(anchors_w_1[:layer_id], dim=1)  # 将anchors_w_1中的元素堆叠起来
                query = query.view(layer_id, self.lstm_size)  # 重塑query的形状
                query = torch.tanh(query + self.w_attn_2(next_h))  # 计算注意力机制的query
                query = self.v_attn(query)  # 计算注意力机制的v
                logits = query.view(1, layer_id)  # 重塑logits的形状
                if self.temperature is not None:  # 如果温度不为空
                    logits /= self.temperature  # 将logits除以温度
                if self.tanh_constant is not None:  # 如果tanh常数不为空
                    logits = self.tanh_constant * torch.tanh(logits)  # 将logits乘以tanh常数
                prob = F.softmax(logits, dim=-1)  # 计算softmax概率
                index = torch.multinomial(prob, 1).long().view(1)  # 从概率分布中采样
                arc_seq.append(index)  # 将索引添加到arc_seq列表中
                arc_seq.append(0)  # 将0添加到arc_seq列表中
                curr_log_prob = F.cross_entropy(logits, index)  # 计算当前对数概率
                log_prob.append(curr_log_prob)  # 将当前对数概率添加到log_prob列表中
                curr_ent = -torch.mean(
                    torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()  # 计算当前熵
                entropy.append(curr_ent)  # 将当前熵添加到entropy列表中
                prev_layers.append(anchors[index])  # 将当前索引对应的anchors添加到prev_layers列表中
                inputs = prev_layers[-1].view(1, -1).requires_grad_()  # 将prev_layers的最后一个元素重塑为输入，并设置为需要梯度

            for i in range(2):  # 遍历两个操作
                embed = inputs  # 获取输入
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))  # 计算下一个LSTM单元的h和c状态
                prev_c, prev_h = next_c, next_h  # 更新前一个LSTM单元的c和h状态
                logits = self.w_soft(next_h) + self.b_soft.requires_grad_()  # 计算softmax的logits
                if self.temperature is not None:  # 如果温度不为空
                    logits /= self.temperature  # 将logits除以温度
                if self.tanh_constant is not None:  # 如果tanh常数不为空
                    op_tanh = self.tanh_constant / self.op_tanh_reduce  # 计算操作tanh
                    logits = op_tanh * torch.tanh(logits)  # 将logits乘以操作tanh
                if use_bias:  # 如果使用偏置
                    logits += self.b_soft_no_learn  # 将不学习的偏置添加到logits
                prob = F.softmax(logits, dim=-1)  # 计算softmax概率
                op_id = torch.multinomial(prob, 1).long().view(1)  # 从概率分布中采样
                arc_seq[2 * i - 3] = op_id  # 将操作ID添加到arc_seq列表中
                curr_log_prob = F.cross_entropy(logits, op_id)  # 计算当前对数概率
                log_prob.append(curr_log_prob)  # 将当前对数概率添加到log_prob列表中
                curr_ent = -torch.mean(
                    torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()  # 计算当前熵
                entropy.append(curr_ent)  # 将当前熵添加到entropy列表中
                inputs = self.encoder(op_id + 1)  # 将操作ID通过嵌入层转换为输入

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))  # 计算下一个LSTM单元的h和c状态
            prev_c, prev_h = next_c, next_h  # 更新前一个LSTM单元的c和h状态
            anchors.append(next_h)  # 将下一层的h状态添加到anchors列表中
            anchors_w_1.append(self.w_attn_1(next_h))  # 计算注意力机制的w_1，添加到anchors_w_1列表中
            inputs = self.encoder(torch.zeros(1).long().cuda())  # 初始化输入，将其通过嵌入层，并移到GPU上
            layer_id += 1  # 增加层ID

        arc_seq = torch.tensor(arc_seq)  # 将arc_seq转换为张量
        entropy = sum(entropy)  # 计算总熵
        log_prob = sum(log_prob)  # 计算总对数概率
        last_c = next_c  # 设置最后一个LSTM单元的c状态
        last_h = next_h  # 设置最后一个LSTM单元的h状态

        return arc_seq, entropy, log_prob, last_c, last_h  # 返回arc_seq, 熵, 对数概率, 最后一个LSTM单元的c状态, 最后一个LSTM单元的h状态
