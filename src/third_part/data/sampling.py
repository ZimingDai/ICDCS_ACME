#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
import random

seed =1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def noniid_share_specified_category(args,dataset, num_users, dataset_name):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index of train, and the labels on clients
    """
    #10个类的IID
    if args.specified_category == 'IID' and args.dataset == "cifar10":
        label_user_groups = {0: [0,3,4,5], 1: [0,3,4,5], 2: [0,3,4,5],
                             3: [0,3,4,5], 4: [0,3,4,5], 5: [0,3,4,5],
                             6: [0,3,4,5],7: [0,3,4,5], 8: [0,3,4,5], 9: [0,3,4,5]}
    # 10个类的C2
    if args.specified_category == 'C2' and args.dataset == "cifar10":
        label_user_groups = {0: [0,3,4,5], 1: [1,2,7,8], 2: [1,2,7,8],
                             3: [0,3,4,5], 4: [1,2,7,8], 5: [1,2,7,8],
                             6: [0,3,4,5],7: [1,2,7,8], 8: [1,2,7,8], 9: [1,2,7,8]}
    # # 10个类的C3
    if args.specified_category == 'C3' and args.dataset == "cifar10":
        label_user_groups = {0: [0, 3, 4, 5], 1: [1, 2, 7, 8], 2: [0, 3, 6, 9],
                             3: [0, 3, 4, 5], 4: [1, 2, 7, 8], 5: [0,3,6,9],
                             6: [0, 3, 4, 5], 7: [1, 2, 7, 8], 8: [0,3,6,9], 9: [1,2,7,8]}

    # 10个类的C4
    if args.specified_category == 'C4' and args.dataset == "cifar10":
        label_user_groups = {0: [0, 3, 4, 5], 1: [1, 2, 7, 8], 2: [0,2,6,9],
                             3: [0, 3, 4, 5], 4: [1, 2, 7, 8], 5: [0,2,6,9],
                             6: [0, 3, 4, 5], 7: [1, 2, 7, 8], 8: [0,2,6,9], 9: [3,7,6,9]}

    #cifar100的non-IID情况
    list1=[23, 4, 7, 27, 89, 48, 9, 15, 66, 94, 36, 39, 60, 13, 50, 29, 38, 8, 52, 18, 90, 21, 81, 62, 99]
    list2=[61,34,14,88,76,67,78,59,20,74,65,2,70,92,97,19,28,83,11,95,32,56,53,17,91]
    list3=[12,85,42,35,64,24,98,25,45,33,10,75,26,1,3,84,87,40,37,30,54,58,16,68,93]
    list4=[79,77,82,72,41,47,31,73,49,55,5,69,86,80,22,57,0,46,43,6,71,51,44,63,96]
    list1_3 = [23, 4, 7, 27, 89, 48, 9, 15, 66, 94, 36, 39, 60,12,85,42,35,64,24,98,25,45,33,10]
    list1_2_3 = [61,34, 14,88, 89, 48, 9, 15, 66, 94, 36, 39, 60,12,85,42,35,64,24,98,56,53,17,91]
    list1_4 = [79,77,82,72,41,47,31,73,49,55,5,69,86,80,50, 29, 38, 8, 52, 18, 90, 21, 81, 62, 99]

    if args.specified_category == 'IID' and args.dataset == "cifar100":
        # label_user_groups = {0: list1, 1: list1, 2: list1,
        #                      3: list1, 4: list1, 5: list1,
        #                      6: list1,7: list1, 8: list1, 9: list1}
        label_user_groups = {0: list1, 1: list1, 2: list1,
                             3: list1, 4: list1}
    # 10个类的C2
    if args.specified_category == 'C2' and args.dataset == "cifar100":
        # label_user_groups = {0: list1, 1: list1, 2: list1,
        #                      3: list2, 4: list2, 5: list2,
        #                      6: list2,7: list2, 8: list2, 9: list2}
        label_user_groups = {0: list1, 1: list1, 2: list1,
                             3: list2, 4: list2}
    # # 10个类的C3
    if args.specified_category == 'C3' and args.dataset == "cifar100":
        # label_user_groups = {0: list1, 1: list2, 2: list1_3,
        #                      3: list1, 4: list2, 5: list1_3,
        #                      6:list1, 7: list2, 8: list1_3, 9: list2}
        label_user_groups = {0: list1, 1: list2, 2: list1_3,
                             3: list1, 4: list2}

    # 10个类的C4
    if args.specified_category == 'C4' and args.dataset == "cifar100":
        # label_user_groups = {0: list1, 1: list2, 2: list1_2_3,
        #                      3: list1, 4: list2, 5: list1_2_3,
        #                      6: list1, 7: list2, 8: list1_2_3, 9: list1_4}
        label_user_groups = {0: list1, 1: list2, 2: list1_2_3,
                             3: list1, 4: list1_4}

    dict_label2client = {i: [] for i in range(args.num_labels)} #key:label, value:client
    dict_users = {i: [] for i in range(num_users)}   #key:client, value:[idxs]
    for i in range(args.num_labels):
        for client in range(len(label_user_groups)):
            labels = label_user_groups[client]
            if i in labels:
                dict_label2client[i].append(client)
    all_idxs = [i for i in range(len(dataset))]
    tracker = split_test_dataset_by_label(args, dataset, all_idxs)  #传进来的是train，不是test

    for label, items in tracker.items():
        if len(dict_label2client[label])!=0:
            user_num = len(dict_label2client[label])
            user_samplenum = int(len(items)*0.1)
            for i in range(user_num):
                dict_users[dict_label2client[label][i]] += list(set(np.random.choice(items, user_samplenum,replace=False)))

    return dict_users,label_user_groups

def data_test(args,dataset, num_users, label_user_groups):
    """
    Return: a dict of image index of test
    """
    all_idxs = [i for i in range(len(dataset))]
    dict_users = {i: [] for i in range(num_users)}

    dict_label2client = {i: [] for i in range(args.num_labels)}
    for i in range(args.num_labels):
        for client in label_user_groups.keys():
            labels = label_user_groups[client]
            if i in labels:
                dict_label2client[i].append(client)

    tracker = split_test_dataset_by_label(args,dataset,all_idxs)
    #客户端测试集
    for label, items in tracker.items():
        if len(dict_label2client[label]) != 0:
            user_num = len(dict_label2client[label])
            user_samplenum = int(len(items)*0.2)    #最原始是测试集每个类1000样本，现在分到每个类分到客户端是200个样本
            for i in range(user_num):
                dict_users[dict_label2client[label][i]] += list(
                    set(np.random.choice(items, user_samplenum, replace=False)))

    return dict_users

def split_test_dataset_by_label(args,data_source,idxs):
    label_num = args.num_labels
    output = {label:[] for label in range(label_num)}
    for idx in idxs:
        item_label = data_source.targets[idx]
        if torch.is_tensor(item_label):
            item_label = item_label.item()
        output[item_label].append(idx)
    return output
