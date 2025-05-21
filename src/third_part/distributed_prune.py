"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import copy

import numpy
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# ++++++for pruning
import os, sys
from tqdm import tqdm
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
from tensorboardX import SummaryWriter

import numpy as np
sys.path.append('/data/gaofei/project/Taylor_pruning-master')
from options import args_parser
from models.network import get_network
from data.data import get_dataset_distributed
from distributed.client import Client
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def average_importances(weights,clients_mask):
    params_num = len(clients_mask[0])
    results = []

    for i,client_mask in enumerate(clients_mask):
        if i==0:
            for j in range(params_num):
                new_value = weights[i]*clients_mask[i][j]
                results.append(new_value)
        else:
            for j in range(params_num):
                results[j] += weights[i]*clients_mask[i][j]
    return results

def normal_matrix(dis_matrix):
    alpha = 9.0
    for k in range(len(dis_matrix)):
        weights = dis_matrix[k]
        for j in range(len(weights)):
            if k==j:
                dis_matrix[k][j]= alpha*dis_matrix[k][j]
            else:
                dis_matrix[k][j] = (10.0-alpha) * dis_matrix[k][j]
    # 使用softmax函数
    exp_sym_dis_matrix = np.exp(dis_matrix)
    row_sums = exp_sym_dis_matrix.sum(axis=1)
    normalized_matrix = exp_sym_dis_matrix / row_sums[:, np.newaxis]

    return normalized_matrix


def distance_importances(args,dis_matrix,clients_mask):
    params_num = len(clients_mask[0])
    final_results = []
    # if args.distance_metric =='emd':
    #     dis_matrix = normal_matrix(dis_matrix)
    # print('ok')
    for k in range(len(dis_matrix)):
        results = []
        weights = dis_matrix[k]
        for i,client_mask in enumerate(clients_mask):
            if i==0:
                for j in range(params_num):
                    new_value = weights[i]*clients_mask[i][j]
                    results.append(new_value)
            else:
                for j in range(params_num):
                    results[j] += weights[i]*clients_mask[i][j]
        final_results.append(results)
    return final_results

def fix_distribution(distribution):
    # Ensure all values are non-negative
    distribution = np.maximum(distribution, 0)
    # Normalize to sum up to 1
    distribution /= np.sum(distribution)
    return distribution

def JS_distance(distribution_p, distribution_q):
    # Ensure the distributions are valid probability distributions
    distribution_p = np.asarray(distribution_p, dtype=np.float64)
    distribution_q = np.asarray(distribution_q, dtype=np.float64)

    distribution_p_smooth = fix_distribution(distribution_p)
    distribution_q_smooth = fix_distribution(distribution_q)

    epsilon = 1e-6
    distribution_p_smooth = distribution_p_smooth + epsilon
    distribution_q_smooth = distribution_q_smooth + epsilon

    # Compute the Jensen-Shannon distance
    M = 0.5 * (distribution_p_smooth + distribution_q_smooth)
    js_distance = 0.5 * (F.kl_div(torch.tensor(distribution_p_smooth).log(), torch.tensor(M), reduction='batchmean') +
                         F.kl_div(torch.tensor(distribution_q_smooth).log(), torch.tensor(M), reduction='batchmean'))
    return emd_similarity(js_distance.item())


def emd_similarity(emd_value):
    """
    将 Earth Mover's Distance (EMD) 转换为相似度分数的函数

    Parameters:
    - emd_value (float): EMD 的数值

    Returns:
    - similarity_score (float): 相似度分数，范围在 [0, 1] 之间，数值越高表示越相似
    """
    similarity_score = 1 / (1 + emd_value)
    return similarity_score

def EMD_distance(distribution_p, distribution_q):
    """
    计算两个数据分布之间的推土机距离（Earth Mover's Distance, EMD）

    参数:
    distribution_p -- 第一个概率分布
    distribution_q -- 第二个概率分布

    返回:
    emd_distance -- 推土机距离
    """
    # Ensure the distributions are valid probability distributions
    distribution_p = np.asarray(distribution_p, dtype=np.float64)
    distribution_q = np.asarray(distribution_q, dtype=np.float64)
    distribution_p_flat = distribution_p.flatten()
    distribution_q_flat = distribution_q.flatten()
    emd_distance = wasserstein_distance(distribution_p_flat, distribution_q_flat)

    return emd_similarity(emd_distance)

def get_distribution(loader):
    model_dir = '/data/gaofei/project/Taylor_pruning-master/dynavit/'
    config = ViTConfig.from_pretrained(model_dir, num_labels=100, output_hidden_states=True)
    model = ViTModel.from_pretrained(model_dir, config=config)
    model.eval()

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
    feature_list = []
    feature_dict ={i:[] for i in range(100)}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # Scale data from [-1, 1] to [0, 1]
            data = (data + 1) / 2

            # Convert to PIL image compatible range [0, 255]
            data = (data * 255).to(torch.uint8)

            # Move data to GPU
            data = data.to(device)

            inputs = feature_extractor(images=data, return_tensors="pt").to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态
            features = hidden_states[:, 0, :]  # 通常使用[CLS] token的特征
            for i in range(len(target)):
                label = target[i].item()
                feature = features[i]
                feature_dict[label].append(feature.cpu().numpy())
    for i in range(100):
        feature_list.extend(feature_dict[i])


    # 计算特征分布，例如使用PCA
    # pca = PCA(n_components=2)  # 将高维特征降到2维以便可视化
    # feature_distribution = pca.fit_transform(feature_matrix)

    return numpy.array(feature_list)
def cal_distances(args,dataset,user_groups,num_users):
    dis_matrix = []
    num_samples = 200  # 只抽取200个数据
    features = []
    for i in range(num_users):
        current_indices = list(user_groups[i])
        current_sampled_indices = random.sample(current_indices, num_samples)  # 随机抽取200个数据
        current_sampler = SubsetRandomSampler(current_sampled_indices)
        current_loader = DataLoader(dataset, batch_size=200, sampler=current_sampler, shuffle=False)
        distribution = get_distribution(current_loader)
        features.append(distribution)
    for i in range(num_users):
        print('--------------------client idx:{}'.format(i))
        current_distribution = features[i]
        row = []
        for j in range(num_users):
            print('--------------------compared client idx:{}'.format(j))
            compared_distribution = features[j]
            if args.distance_metric == 'js':
                distance = JS_distance(current_distribution, compared_distribution)
            elif args.distance_metric == 'emd':
                distance = EMD_distance(current_distribution, compared_distribution)
            row.append(distance)
        dis_matrix.append(row)
    dis_matrix = np.array(dis_matrix)
    sym_dis_matrix = np.sqrt(dis_matrix * dis_matrix.T)
    # 使用softmax函数
    exp_sym_dis_matrix = np.exp(sym_dis_matrix)
    row_sums = exp_sym_dis_matrix.sum(axis=1)
    normalized_matrix = exp_sym_dis_matrix / row_sums[:, np.newaxis]
    dis_matrix = normalized_matrix
    return dis_matrix

def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    # dataset loading section
    train_dataset, test_dataset, user_groups, test_user_groups = get_dataset_distributed(args)
    # 依据客户端数据计算对应的距离
    if args.approach == 'pruning_per' and args.distance_metric!='avg':
        dis_matrix = cal_distances(args,train_dataset,user_groups,args.num_users)

    model = get_network(args)

    clients_data_ratio = []
    all_data_num = 0
    for i in range(len(user_groups)):
        all_data_num += len(user_groups[i])

    for i in range(len(user_groups)):
        clients_data_ratio.append(len(user_groups[i]) / all_data_num)

    log_save_folder = "%s" % args.name
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)

    train_writer = SummaryWriter(logdir="%s" % (log_save_folder))

    # 初始化客户端
    clients = []
    for client_idx in range(args.num_users):
        client = Client(args, copy.deepcopy(model), train_dataset, test_dataset, user_groups[client_idx], test_user_groups[client_idx], client_idx,train_writer)
        client.prepare_model()
        client.prepare_pruning()
        clients.append(client)

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Epoch : {epoch + 1} |\n')

        if args.approach =='wo_pruning':
            for client_idx in range(args.num_users):
                _, _, _, _ = clients[client_idx].local_training_alone(epoch)
        elif args.approach =='pruning_alone':
            for client_idx in range(args.num_users):
                client = clients[client_idx]
                importances, _, _ = client.local_training(epoch)
                client.pruning_model(importances)
                prec1, _ = client.validate(client.args, client.test_loader, client.local_model, client.device, client.criterion,
                                         epoch, train_writer=client.train_writer)
        elif args.approach == 'pruning_avg':
            clients_importance = []
            for client_idx in range(args.num_users):
                importances,_,_ = clients[client_idx].local_training(epoch)
                clients_importance.append(importances)
            weighted_importances = average_importances(clients_data_ratio, clients_importance)
            for client_idx in range(args.num_users):
                client = clients[client_idx]
                client.pruning_model(weighted_importances)
                client.validate(client.args, client.test_loader, client.local_model, client.device, client.criterion,
                                         epoch, train_writer=client.train_writer)
        else:
            clients_importance = []
            for client_idx in range(args.num_users):
                importances, _, _ = clients[client_idx].local_training(epoch)
                clients_importance.append(importances)
            weighted_importances_list = distance_importances(args,dis_matrix, clients_importance)
            for client_idx in range(args.num_users):
                client = clients[client_idx]
                client.pruning_model(weighted_importances_list[client_idx])
                client.validate(client.args, client.test_loader, client.local_model, client.device, client.criterion,
                                epoch, train_writer=client.train_writer)


if __name__ == '__main__':
    main()
