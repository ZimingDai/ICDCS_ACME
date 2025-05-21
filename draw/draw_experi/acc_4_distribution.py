# -*- coding: utf-8 -*-
"""
@File    :   acc_4_distribution
@Time    :   2025/5/21 16:42
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   不同数据分布下的acc提升
"""
import matplotlib.pyplot as plt

if __name__ == "__main__":


    # 方法和差值数据
    methods = ['Alone', 'Avg', 'JS', 'Ours']
    values = [1.8, 0.6, 2.6, 3.8]
    # iid: [2, 3.8, 3.4, 4]
    # c1: [1.9, 2.4, 3, 3.8]
    # c2: [2.0, 1.4, 3.2, 3.8]
    # c3: [1.8, 0.6, 2.6, 3.8]

    # 使用从上传的图片中提取的颜色方案
    colors = ['#374f99', '#6ea4c4', '#c6e5ea', '#f1b069']

    # 定义不同的hatch样式
    hatch_patterns = ['/', '\\', '|', '-']

    # 创建柱状图
    plt.figure(figsize=(5.5, 5))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', linewidth=2.5)

    # 应用不同的hatch样式到每个柱子
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)

    # 设置标签和字体大小
    # plt.ylabel('Difference in Accuracy (%)', fontsize=24)

    # 移除横坐标的标签
    plt.xticks([])

    # 设置纵坐标的文字大小
    plt.yticks(ticks=[0, 1, 2, 3, 4, 5], fontsize=24)

    # 保存图形
    # plt.savefig('figs/emd_acc_c3.png', dpi=300, bbox_inches='tight', transparent=True)

    # 显示图形
    plt.show()

    # C2:emd 75.4 js 74.6 alone 74.8 avg 74.4原始:70.2

    # C2
    methods = ['Alone', 'Avg', 'JS', 'Ours']
    values = [74.8 - 70.2, 74.4 - 70.2, 74.6 - 70.2, 75.4 - 70.2]

    # 使用从上传的图片中提取的颜色方案
    colors = ['#374f99', '#6ea4c4', '#c6e5ea', '#f1b069']

    # 定义不同的hatch样式
    hatch_patterns = ['/', '\\', '|', '-']

    # 创建柱状图
    plt.figure(figsize=(5.5, 5))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', linewidth=2.5)

    # 应用不同的hatch样式到每个柱子
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)

    # 设置标签和字体大小
    # plt.ylabel('Difference in Accuracy (%)', fontsize=24)

    # 移除横坐标的标签
    plt.xticks([])

    # 设置纵坐标的文字大小
    plt.yticks(fontsize=20)

    # 特别设置'Ours'标签颜色为红色
    for label in plt.gca().get_xticklabels():
        if label.get_text() == 'Ours':
            label.set_color('red')

    # 保存图形
    plt.savefig('figs/emd_acc_c2.png', dpi=300, bbox_inches='tight', transparent=True)

    # 显示图形
    plt.show()

    # C3:emd 72.2 js 72 alone 71.6 avg 71.8原始:69.6
    # C3
    methods = ['Alone', 'Avg', 'JS', 'Ours']
    values = [71.6 - 69.6, 71.8 - 69.6, 72 - 69.6, 72.2 - 69.6]

    # 使用从上传的图片中提取的颜色方案
    colors = ['#374f99', '#6ea4c4', '#c6e5ea', '#f1b069']

    # 定义不同的hatch样式
    hatch_patterns = ['/', '\\', '|', '-']

    # 创建柱状图
    plt.figure(figsize=(5.5, 5))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', linewidth=2.5)

    # 应用不同的hatch样式到每个柱子
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)

    # 设置标签和字体大小
    # plt.ylabel('Difference in Accuracy (%)', fontsize=24)

    # 移除横坐标的标签
    plt.xticks([])

    # 设置纵坐标的文字大小
    plt.yticks(ticks=[0, 1, 2, 3], fontsize=20)

    # 保存图形
    plt.savefig('figs/emd_acc_c3.png', dpi=300, bbox_inches='tight', transparent=True)

    # 显示图形
    plt.show()

    # C4:emd 72.8 js 72.2 alone 72.2 avg 71.6原始:69.6
    # C4
    methods = ['Alone', 'Avg', 'JS', 'Ours']
    values = [72.2 - 69.6, 71.6 - 69.6, 72.2 - 69.6, 72.8 - 69.6]

    # 使用从上传的图片中提取的颜色方案
    colors = ['#374f99', '#6ea4c4', '#c6e5ea', '#f1b069']

    # 定义不同的hatch样式
    hatch_patterns = ['/', '\\', '|', '-']

    # 创建柱状图
    plt.figure(figsize=(5.5, 5))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', linewidth=2.5)

    # 应用不同的hatch样式到每个柱子
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)

    # 设置标签和字体大小
    # plt.ylabel('Difference in Accuracy (%)', fontsize=24)

    # 移除横坐标的标签
    plt.xticks([])

    # 设置纵坐标的文字大小
    plt.yticks(ticks=[0, 1, 2, 3], fontsize=20)

    # 保存图形
    # plt.savefig('figs/emd_acc_c4.png', dpi=300, bbox_inches='tight', transparent=True)

    # 显示图形
    plt.show()
