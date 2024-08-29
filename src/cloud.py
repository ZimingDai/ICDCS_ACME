#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cloud.py
@Time    :   2024/06/26 09:32:41
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   云服务器运行的生成DynaViTw和DynaViT
"""

from __future__ import absolute_import, division, print_function
import argparse
from cloud_func import *
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor


def get_args():
    parser = argparse.ArgumentParser()

    # 数据相关参数
    parser.add_argument("--data_name", default=None, type=str, required=True,
                        help="The input data . Should contain the dataset for the task.")
    parser.add_argument("--output_dir", default="../model/dynavitw/", type=str,
                        help="The output directory where the trained model will be saved.")

    # 模型相关参数
    parser.add_argument("--model_dir", default="../vit-base-patch16-224", type=str,
                        help="The pretrained model directory.")
    # parser.add_argument("--model_type", default="vit", type=str, required=True,
    #                     help="Model type selected in the list: vit.")

    # 任务相关参数
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")

    # 训练参数
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if applied.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for initialization")

    # ViT特定参数
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate on hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate on attention probabilities.")

    # 数据增强参数
    parser.add_argument('--data_aug', action='store_true', help="Whether to use data augmentation")

    # Depth方向参数
    parser.add_argument('--depth_mult_list', type=str, default='1.',
                        help="The possible depths used for training, e.g., '1.' is for default")
    parser.add_argument("--depth_lambda1", default=1.0, type=float,
                        help="Logit matching coefficient.")
    parser.add_argument("--depth_lambda2", default=1.0, type=float,
                        help="Hidden states matching coefficient.")

    # Width方向参数
    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="The possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")
    parser.add_argument("--width_lambda1", default=1.0, type=float,
                        help="Logit matching coefficient.")
    parser.add_argument("--width_lambda2", default=0.1, type=float,
                        help="Hidden states matching coefficient.")

    parser.add_argument("--training_phase", default="dynavitw", type=str,
                        help="Can be finetuning, dynabertw, dynabert, final_finetuning")
    parser.add_argument("--device", default=1, type=str, help="Choose the device to train.")

    parser.add_argument("--output_mode", default="classification", type=str, help="Task Type")

    args = parser.parse_args()

    return args


# 主函数
def main(file_dir, args):
    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]

    # 设置CUDA、GPU和分布式训练

    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{args.device}")
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available, using CPU instead.")
        current_device = 'cpu'

    # 定义使用的设备
    device = torch.device(current_device)

    args.n_gpu = 1
    args.device = device

    # 设置随机种子
    set_seed(args)

    # 准备CIFAR-100任务：在此提供num_labels
    args.task_name = args.task_name.lower()
    if args.task_name == 'cifar100':
        num_labels = 100
    elif args.task_name == 'cifar10':
        num_labels = 10
    elif args.task_name == 'stanford_car':
        num_labels = 196

    # 准备模型、特征提取器和配置
    config = ViTConfig.from_pretrained(args.model_dir, num_labels=num_labels, output_hidden_states=True)
    feature_extractor = ViTImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir, config=config)

    # 如果需要，加载教师模型
    if args.training_phase == 'dynavitw' or args.training_phase == 'dynavit':
        teacher_model = ViTForImageClassification.from_pretrained(args.model_dir, config=config)
        teacher_model.to(args.device)
    else:
        teacher_model = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.training_phase == 'dynavitw':
        print("********begin to prune the width********")
        # rewire the network according to the importance of attention heads and neurons
        head_importance, neuron_importance = compute_neuron_head_importance(args, model, feature_extractor)
        reorder_neuoron_and_head(model, head_importance, neuron_importance)

    model.to(args.device)

    logger.info("训练/评估参数 %s", args)

    # 训练
    if args.do_train:
        print("********begin to train********")
        train_dataset = load_and_cache_examples(feature_extractor, args.data_name, evaluate=False)
        if teacher_model:
            global_step, tr_loss = train(args, train_dataset, model, feature_extractor, teacher_model, file_dir)
        else:
            global_step, tr_loss = train(args, train_dataset, model, feature_extractor, None, file_dir)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    print("Program Finished!")


# 如果该脚本作为主程序运行，调用main函数
if __name__ == "__main__":
    args = get_args()

    FILE_DIR = f"../log/{args.task_name}.txt"

    with open(FILE_DIR, 'a') as file_dir:
        for arg in vars(args):
            print(f"--{arg}: {getattr(args, arg)}", file=file_dir)
        main(file_dir, args)
