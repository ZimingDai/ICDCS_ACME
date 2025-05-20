#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   train_search.py
@Time    :   2024/06/14 13:36:24
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   使用NAS进行生成Coarse Header。
"""
import argparse
import glob
import logging
import os
import random
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造目标模块的路径并添加到sys.path
target_dir = os.path.abspath(os.path.join(current_dir, '../data'))
sys.path.append(target_dir)

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from tqdm import tqdm
import utils
from data import get_loaders
from micro_controller import Controller
from micro_child import ViTWithENASCNN
from transformers import ViTConfig

parser = argparse.ArgumentParser("cifar")
# 添加数据集和训练参数
parser.add_argument('--data', type=str, default='../data', help='数据集位置')
parser.add_argument('--dataset_name', type=str, default='CIFAR100', help='数据集种类:CIFAR10,CIFAR100,stanford_car')

parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--report_freq', type=float, default=10, help='报告频率')
parser.add_argument('--gpu', type=int, default=1, help='GPU设备ID, 多个GPU用逗号分隔')
parser.add_argument('--epochs', type=int, default=130, help='训练轮数')
parser.add_argument('--model_path', type=str, default='../model', help='模型保存路径')
parser.add_argument('--save', type=str, default='EXP', help='实验名称')
parser.add_argument('--seed', type=int, default=0, help='随机种子')

# 添加神经网络参数
parser.add_argument('--child_lr_max', type=float, default=0.1, help='子网络最大学习率')
parser.add_argument('--child_lr_min', type=float, default=0.0005, help='子网络最小学习率')
parser.add_argument('--child_lr_T_0', type=int, default=10, help='子网络初始学习率周期')
parser.add_argument('--child_lr_T_mul', type=int, default=2, help='子网络学习率周期倍增')
parser.add_argument('--child_num_layers', type=int, default=3, help='子网络层数')
parser.add_argument('--child_out_filters', type=int, default=256, help='子网络输出过滤器数')
parser.add_argument('--child_num_branches', type=int, default=6, help='子网络分支数')
parser.add_argument('--child_num_cells', type=int, default=3
                    , help='子网络单元数')
parser.add_argument('--child_use_aux_heads', type=bool, default=False, help='是否使用辅助头')

# 添加控制器参数
parser.add_argument('--controller_lr', type=float, default=0.0025, help='控制器学习率')
parser.add_argument('--controller_tanh_constant', type=float, default=1.10, help='控制器tanh常数')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='控制器操作tanh减少系数')

# 添加LSTM参数
parser.add_argument('--lstm_size', type=int, default=100, help='LSTM大小')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='LSTM层数')
parser.add_argument('--lstm_keep_prob', type=float, default=0, help='LSTM保持概率')
parser.add_argument('--temperature', type=float, default=5.0, help='温度')

# 添加熵权重参数
parser.add_argument('--entropy_weight', type=float, default=0.0001, help='熵权重')
parser.add_argument('--bl_dec', type=float, default=0.99, help='基线衰减')

# 添加微调参数
parser.add_argument('--finetune_epochs', type=int, default=50, help='微调训练轮数')

# 添加ViT参数
parser.add_argument('--vit_model_path', type=str, default='../model/vit_cifar100_model', help='ViT模型读取路径')

# 添加微调参数
parser.add_argument('--width_mult', type=float, default=0.75, help='backbone 宽度')
parser.add_argument('--depth_mult', type=float, default=0.5, help='backbone 深度')

parser.add_argument('--save_path', type=str, default='../log/NAS_part/', help='log保存路径')

args = parser.parse_args()

# 设置实验保存路径
args.save = args.save_path + '/{}-{}'.format(args.save, time.strftime(
    "%Y.%m.%d-%H:%M:%S") + f'-{args.child_num_cells}-{args.child_num_layers}-{args.width_mult}-{args.depth_mult}')
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('train_search.py'))

# 设置日志格式
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset_name == "CIFAR100":
    CIFAR_CLASSES = 100
elif args.dataset_name == 'CIFAR10':
    CIFAR_CLASSES = 10
elif args.dataset_name == 'stanford_car':
    CIFAR_CLASSES = 196
baseline = None
epoch = 0
best_dag = None


def main():
    global best_dag
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    config = ViTConfig.from_pretrained(args.vit_model_path, num_labels=CIFAR_CLASSES)
    model = ViTWithENASCNN.from_pretrained(args.vit_model_path, config=config, args=args, num_classes=CIFAR_CLASSES)
    model.cuda()

    controller = Controller(args)
    controller.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.controller_lr,
        betas=(0.1, 0.999),
        eps=1e-3,
    )

    train_loader, reward_loader, valid_loader = get_loaders(args)

    scheduler = utils.LRScheduler(optimizer, args)

    model.apply(lambda m: setattr(m, 'depth_mult', args.depth_mult))
    model.apply(lambda m: setattr(m, 'width_mult', args.width_mult))

    best_train = (None, 0)

    for epoch in tqdm(range(args.epochs), desc='Epoch: '):
        lr = scheduler.update(epoch)
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_dag = train(train_loader, model, controller, optimizer)
        logging.info('train_acc %f', train_acc)

        if train_acc > best_train[1]:
            best_train = (train_dag, train_acc)

        train_controller(reward_loader, model, controller, controller_optimizer)

        # valid_acc, dag = infer(valid_loader, model, controller)
        valid_acc = validate(valid_loader, model, train_dag)
        logging.info('valid_acc %f', valid_acc)

        if best_dag is None or valid_acc > best_dag[1]:
            best_dag = (train_dag, valid_acc)
            utils.save(model, os.path.join(args.save, 'weights.pt'))

    # logging.info('Finetuning the best architecture...')
    # finetune(best_dag[0], model, train_loader, valid_loader)

    finetune(best_dag[0], model, train_loader, valid_loader)

    logging.info('best_dag is %s and %f', str(best_dag[0]), best_dag[1])
    logging.info('best_train is %s and %f', str(best_train[0]), best_train[1])


def train(train_loader, model, controller, optimizer):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    final_dag = None

    best_acc = 0

    for step, (data, target) in enumerate(train_loader):
        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        controller.eval()
        dag, _, _ = controller()

        # 仅传递必须的参数，确保没有传递无效参数
        output = model(pixel_values=data, dag=dag)  # 修改此行，确保传递的参数名称正确

        logits = output.logits
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, total_loss.avg, total_top1.avg)
        if total_top1.avg > best_acc:
            best_acc = total_top1.avg
            final_dag = dag
    logging.info('train_final_best %s %f', str(final_dag), best_acc)
    return total_top1.avg, final_dag


def train_controller(reward_loader, model, controller, controller_optimizer):
    global baseline
    total_loss = utils.AvgrageMeter()
    total_reward = utils.AvgrageMeter()
    total_entropy = utils.AvgrageMeter()

    for step in range(300):
        data, target = reward_loader.next_batch()
        model.eval()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        controller.train()
        dag, log_prob, entropy = controller()

        with torch.no_grad():
            output = model(pixel_values=data, dag=dag)
            logits = output.logits  # 修改此行，确保传递的参数名称正确
            reward = utils.accuracy(logits, target)

        if args.entropy_weight is not None:
            reward += args.entropy_weight * entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = log_prob * (reward - baseline)
        loss = loss.sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)

        if step % args.report_freq == 0:
            logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, baseline.item())


def infer(valid_loader, model, controller):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()
    controller.eval()

    best_valid_acc = 0
    best_dag = None

    with torch.no_grad():
        for step in range(10):
            data, target = valid_loader.next_batch()
            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            output = model(pixel_values=data, dag=dag)
            logits = output.logits  # 修改此行，确保传递的参数名称正确
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            logging.info('valid %03d %e %f', step, loss.item(), prec1.item())
            logging.info('normal cell %s', str(dag))

            if prec1.item() > best_valid_acc:
                best_valid_acc = prec1.item()
                best_dag = dag

    return total_top1.avg, best_dag


def finetune(dag, model, train_loader, valid_loader):
    # config = ViTConfig.from_pretrained(args.vit_model_path)
    # model = ViTWithENASCNN(config, args)
    # model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpu.split(',')])  # 修改此行，支持多GPU
    # model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = utils.LRScheduler(optimizer, args)

    for epoch in tqdm(range(args.finetune_epochs), desc='Finetune Epoch: '):
        lr = scheduler.update(epoch)
        logging.info('finetune epoch %d lr %e', epoch, lr)

        total_loss = utils.AvgrageMeter()
        total_top1 = utils.AvgrageMeter()

        for step, (data, target) in enumerate(train_loader):
            model.train()
            n = data.size(0)

            data = data.cuda()
            target = target.cuda()

            optimizer.zero_grad()

            output = model(pixel_values=data, dag=dag)
            logits = output.logits  # 修改此行，确保传递的参数名称正确
            loss = F.cross_entropy(logits, target)

            loss.backward()
            optimizer.step()

            prec1 = utils.accuracy(logits, target)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('finetune train %03d %e %f', step, total_loss.avg, total_top1.avg)

        logging.info('finetune train_acc %f', total_top1.avg)

        valid_acc = validate(valid_loader, model, dag)
        logging.info('finetune valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'finetuned_weights.pt'))


def validate(valid_loader, model, dag):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step in tqdm(range(len(valid_loader.data_loader)), desc='Validating'):
            data, target = valid_loader.next_batch()

            data = data.cuda()
            target = target.cuda()

            output = model(pixel_values=data, dag=dag)
            logits = output.logits  # 修改此行，确保传递的参数名称正确
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, total_loss.avg, total_top1.avg)

    return total_top1.avg


if __name__ == '__main__':
    main()
