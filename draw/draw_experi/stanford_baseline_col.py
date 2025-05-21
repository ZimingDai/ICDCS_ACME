# -*- coding: utf-8 -*-
"""
@File    :   stanford_baseline_col
@Time    :   2025/5/21 16:51
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   Stanford Car上的baseline柱状图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # 创建数据集
    data = {
        # "Name": ["MobileViT", "Efficient-ViT", "ResMLP-15", "DeViT", "DeDeiTs", "DeCCTs", "Ours(Best)"],
        # "Parameters": [17.43, 5.0, 15, 23, 23.9, 18.3, 20.4],
        # "Accuracy": [86.55, 89.87, 84.6, 89.11, 91.53, 90.67, 92.9755]
        "Name": ["Efficient-ViT", "MobileViT", "ResMLP-15", "Twins-SVT", "DeViT", "DeDeiTs", "DeCCTs", "Ours(Best)"],
        "Parameters": [12.4, 17.43, 15, 24, 23, 23.9, 18.3, 20.4],
        "Accuracy": [0.897, 0.8655, 0.846, 0.9108, 0.8911, 0.9153, 0.9067, 0.929755]
    }

    df = pd.DataFrame(data)
    hatch_patterns = ['/', '-']
    # 设置柱状图的宽度和位置
    bar_width = 0.42
    index = np.arange(len(df['Name']))

    # 创建图形对象
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制参数的柱状图
    bars1 = ax1.bar(index - bar_width / 2, df['Parameters'], bar_width, label='Parameters', color='#6caa8a',
                    hatch=hatch_patterns[0], edgecolor='white')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制准确率的柱状图
    bars2 = ax2.bar(index + bar_width / 2, df['Accuracy'], bar_width, label='Accuracy', color='#3a5171',
                    hatch=hatch_patterns[1], edgecolor='white')

    # 设置准确率的范围
    ax2.set_ylim(0.7, 1)

    # 设置网格
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置标签和标题
    ax1.set_ylabel('Parameter Count(M)', fontsize=26)
    # ax2.set_ylabel('Accuracy', fontsize=24)
    ax1.set_xticks(index)  # 删除横坐标刻度
    ax1.set_xticklabels(df['Name'], rotation=25, ha='right', fontsize=23)

    # 设置纵坐标刻度字体大小
    ax1.tick_params(axis='y', labelsize=25)
    ax2.tick_params(axis='y', labelsize=25)

    # 设置准确率的y_ticks为小数点后一位
    accuracy_ticks = np.round(np.arange(0.70, 1.0, 0.05), 1)
    ax2.set_yticks(accuracy_ticks)

    # 设置横坐标颜色
    colors = ['black'] * len(df['Name'])
    colors[-1] = 'red'
    for ticklabel, tickcolor in zip(ax1.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # 显示图例
    # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=20, framealpha=0.4)

    # 布局调整
    plt.tight_layout()
    plt.savefig('./figs/stan_baseline.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
