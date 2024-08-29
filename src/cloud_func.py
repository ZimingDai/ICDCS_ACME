#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cloud_func.py
@Time    :   2024/06/26 09:33:17
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   云服务器上所需要的函数定义
"""
from __future__ import absolute_import, division, print_function

import random
import torch
import logging
import os
from tqdm import tqdm, trange
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"


# 定义软交叉熵损失函数
def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)  # 计算预测的对数概率
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)  # 计算目标的概率
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()  # 计算软交叉熵损失并返回平均值


# 均方误差损失函数
loss_mse = nn.MSELoss()


# 设置随机种子，保证实验可复现
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# 训练函数
def train(args, train_dataset, model, feature_extractor, teacher_model, file_dir):
    """ 训练模型 """

    # 计算每次训练的批量大小
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # 创建数据采样器，用于随机选择数据
    train_sampler = RandomSampler(train_dataset)

    # 创建数据加载器，用于加载训练数据
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # 计算总的训练步数
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 定义不参与权重衰减的参数
    no_decay = ['bias', 'LayerNorm.weight']

    # 组织模型参数，区分需要权重衰减的和不需要的
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 初始化优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    # 初始化学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # 初始化全局步数和总损失
    global_step = 0
    tr_loss = 0.0

    # 初始化模型梯度
    model.zero_grad()

    # 创建训练迭代器
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    # 设置随机种子
    set_seed(args)

    # 初始化最佳性能评分
    current_best = 0

    # 定义输出文件路径
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')

    # 开始训练迭代
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        # 遍历数据批次
        for step, batch in enumerate(epoch_iterator):
            model.train()  # 设置模型为训练模式
            batch = tuple(t.to(args.device) for t in batch)  # 将数据移至设备

            # 准备输入数据
            inputs = {
                'pixel_values': batch[0],
                'labels': batch[1],
            }

            # 如果处于动态宽度训练阶段且存在教师模型，则准备教师模型的输出
            if args.training_phase == 'dynavitw' and teacher_model:
                with torch.no_grad():
                    teacher_output = teacher_model(**inputs)
                    teacher_logit = teacher_output.logits
                    teacher_reps = teacher_output.hidden_states


            elif args.training_phase == 'dynavit' and teacher_model:
                hidden_max_all, logits_max_all = [], []
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    with torch.no_grad():
                        teacher_output = teacher_model(**inputs)
                        teacher_logit = teacher_output.logits
                        teacher_reps = teacher_output.hidden_states

                        hidden_max_all.append(teacher_reps)
                        logits_max_all.append(teacher_logit)

            # 遍历所有子网络的深度
            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))

                # 调整宽度
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))

                    # 宽度自适应阶段
                    if args.training_phase == 'dynavitw':
                        if getattr(args, 'data_aug'):

                            student_output = model(**inputs)
                            student_logit = student_output.logits
                            student_reps = student_output.hidden_states

                            # 计算logits的蒸馏损失
                            if args.output_mode == "classification":
                                logit_loss = soft_cross_entropy(student_logit, teacher_logit.detach())
                            elif args.output_mode == "regression":
                                logit_loss = 0

                            # 计算隐藏状态的蒸馏损失
                            rep_loss = 0
                            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                                tmp_loss = loss_mse(student_rep, teacher_rep.detach())
                                rep_loss += tmp_loss

                            loss = args.width_lambda1 * logit_loss + args.width_lambda2 * rep_loss
                        else:
                            loss = model(**inputs).loss
                    else:
                        loss = model(**inputs).loss  # 最终微调阶段

                    print("loss: ", loss)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps  # 根据累积步数调整损失

                    loss.backward()  # 反向传播计算梯度

            # 剪裁所有宽度累积的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 更新模型参数
                scheduler.step()  # 更新学习率
                model.zero_grad()  # 清零梯度
                global_step += 1  # 更新步数

                # 定期评估模型性能
                if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        acc = []

                        # 收集所有子网络的性能
                        for depth_mult in sorted(args.depth_mult_list, reverse=True):
                            model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                            for width_mult in sorted(args.width_mult_list, reverse=True):
                                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                                results = evaluate(args, model, feature_extractor)

                                logger.info("********** start evaluate results *********")
                                logger.info("depth_mult: %s ", depth_mult)
                                logger.info("width_mult: %s ", width_mult)
                                logger.info("results: %s ", results)
                                logger.info("********** end evaluate results *********")

                                print("********** start evaluate results *********", file=file_dir)
                                print(f"depth_mult: {depth_mult}", file=file_dir)
                                print(f"width_mult: {width_mult}", file=file_dir)
                                print(f"results: {results}", file=file_dir)
                                print("********** end evaluate results *********", file=file_dir)

                                acc.append(list(results.values())[0])

                        # 如果当前性能超过之前的最佳性能，则保存模型
                        if sum(acc) > current_best:
                            current_best = sum(acc)
                            print("***best***{}\n".format(acc), file=file_dir)
                            with open(output_eval_file, "a") as writer:
                                writer.write("{}\n".format(acc))

                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(args.output_dir, safe_serialization=False)
                            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                            model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
                            feature_extractor.save_pretrained(args.output_dir)

            # 如果完成了所有训练步骤，结束训练
            if 0 < t_total < global_step:
                epoch_iterator.close()
                break

        if 0 < t_total < global_step:
            train_iterator.close()
            break

    # 返回全局步数和平均损失
    print("train FINISH!")
    return global_step, tr_loss / global_step


# 评估函数
def evaluate(args, model, feature_extractor, prefix=""):
    """ 评估模型 """
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(feature_extractor, args.data_name, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating(evaluate)"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'pixel_values': batch[0], 'labels': batch[1]}
                outputs = model(**inputs)

                tmp_eval_loss, logits = outputs.loss, outputs.logits

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        result = (preds == out_label_ids).mean()
        results.update({eval_task: result})

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")  # 将所有结果写入同一个文件
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
            writer.write("\n")
    print("evaluate FINISH!")
    return results


# 加载和缓存数据集
def load_and_cache_examples(feature_extractor, data_name, evaluate=False, ):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
    if data_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root="../data", train=not evaluate, download=False, transform=transform)
    elif data_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root="../data", train=not evaluate, download=False, transform=transform)
    elif data_name == 'stanford_car':
        if evaluate:
            dataset = datasets.StanfordCars(root='../data', split='test', download=False, transform=transform)
        else:
            dataset = datasets.StanfordCars(root='../data', split='train', download=False, transform=transform)
    return dataset


def compute_neuron_head_importance(args, model, feature_extractor):
    # Ensure model is on the right device
    model = model.to(args.device)

    # Prepare things for heads
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # Collect weights
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
        neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    # Evaluation dataset
    eval_dataset = load_and_cache_examples(feature_extractor, args.data_name, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    for batch in tqdm(eval_dataloader, desc="Evaluating(compute_neuron_head_importance)"):
        batch = tuple(t.to(args.device) for t in batch)
        pixel_values, labels = batch

        # Calculate head importance
        outputs = model(pixel_values=pixel_values, head_mask=head_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

        # Calculate neuron importance
        for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight,
                                                  neuron_importance):
            current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
            current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    print("compute_neuron_head_importance FINISH!")
    return head_importance, neuron_importance


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

    print("reorder_neuoron_and_head FINISH!")
