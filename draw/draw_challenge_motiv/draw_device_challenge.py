#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   draw_device_challenge.py
@Time    :   2025/5/21 16:22
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   对Challenge图中的性能异构性画图
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    # 原始数据
    x_labels = ['1', '2', '3', '4', '5']
    group1_values = [2, 6, 5, 9, 2]
    group2_values = [10, 8, 2, 4, 3]

    # 根据 Group 2 的值降序排序
    sorted_indices = np.argsort(group2_values)[::-1]  # 由高到低排序
    x_labels_sorted = x_labels
    group1_values_sorted = [group1_values[i] for i in sorted_indices]
    group2_values_sorted = [group2_values[i] for i in sorted_indices]

    x = np.arange(len(x_labels_sorted))  # 横轴刻度位置
    width = 0.4  # 柱状宽度

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 4))

    # 绘制两组柱状图
    rects1 = ax.bar(x - width / 2, group1_values_sorted, width, label='Group 1',
                    hatch='/', color='#4D7376', edgecolor='white')
    rects2 = ax.bar(x + width / 2, group2_values_sorted, width, label='Group 2',
                    hatch='\\', color='#BA7934', edgecolor='white')

    # 设置横轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels_sorted, fontsize=40)

    ax.tick_params(axis='both', labelsize=40)

    # 优化图形布局
    fig.tight_layout()
    # plt.savefig('a.svg', dpi=300, bbox_inches='tight', transparent=True)
    # 显示图形
    plt.show()
