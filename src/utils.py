#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/06/14 13:37:04
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   一些函数的定义
"""
import os
import math
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class LRScheduler:
    def __init__(self, optimizer, args):
        self.last_lr_reset = 0
        self.lr_T_0 = args.child_lr_T_0
        self.child_lr_T_mul = args.child_lr_T_mul
        self.child_lr_min = args.child_lr_min
        self.child_lr_max = args.child_lr_max
        self.optimizer = optimizer

    def update(self, epoch):
        T_curr = epoch - self.last_lr_reset
        if T_curr == self.lr_T_0:
            self.last_lr_reset = epoch
            self.lr_T_0 = self.lr_T_0 * self.child_lr_T_mul
        rate = T_curr / self.lr_T_0 * math.pi
        lr = self.child_lr_min + 0.5 * (self.child_lr_max - self.child_lr_min) * (1.0 + math.cos(rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = 128

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def accuracy(logits, target):
    """
    计算给定 logits 和 target 的准确率。
    参数:
        logits: 模型的输出，形状为 (batch_size, num_classes)
        target: 真实的标签，形状为 (batch_size)
    返回:
        tuple: 包含 (top1_accuracy, top5_accuracy) 的元组，其中 top1_accuracy 为 top-1 准确率，top5_accuracy 为 top-5 准确率。
    """
    with torch.no_grad():
        # 获取最高 logit 的索引 (即预测的类别)
        _, pred = torch.max(logits, 1)
        # 计算正确预测的数量
        correct = pred.eq(target).float().sum(0)
        # 计算 top-1 准确率
        top1_accuracy = 100.0 * correct / target.size(0)

    return top1_accuracy


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
