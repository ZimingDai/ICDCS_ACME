# -*- coding: utf-8 -*-
"""
@File    :   draw_second_motiv
@Time    :   2025/5/21 16:35
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   Motivation的第二个图
"""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    plt.rcParams['font.family'] = 'Arial'
    # 数据准备
    model_ids = np.array([1, 2, 3, 4, 5, 6])  # 模型编号
    accuracies = np.array([85.70, 86.56, 87.85, 88.52, 89.55, 90.60]) * 0.01  # 准确率数据

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制柱状图
    bars = ax.bar(model_ids, accuracies, color='#6ea4c4', edgecolor='black', linewidth=2.5)

    # 设置图表标题和轴标签
    ax.set_xlabel('Model Index', fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=24)

    # 设置坐标轴刻度大小
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # 设置纵轴的范围以便更好地显示数据
    ax.set_ylim(0.8, 0.95)  # 设置y轴的显示范围
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')

    # 添加数值标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=18)
    plt.savefig(r"motiv_2.png", dpi=300, bbox_inches='tight', transparent=True)
    # 显示图表
    plt.show()
