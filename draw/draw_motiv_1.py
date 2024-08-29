import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Arial'

# 数据准备
classifiers = [i + 1 for i in range(12)]  # 修改为从1开始
accuracies = [1.0, 30.49, 48.56, 62.48, 69.9, 74.83, 77.65, 80.53, 82.54, 84.26, 85.77, 86.1]
energy_ratios = [0.04, 0.09, 0.15, 0.22, 0.29, 0.37, 0.46, 0.55, 0.65, 0.76, 0.88, 1.00]  # 能耗比例

# 色彩调整
deep_red = [139/255, 0, 0]  # 转换RGB值
deep_blue = [0, 0, 139/255]  # 转换RGB值

# 创建图形和轴对象
fig, ax1 = plt.subplots(figsize=(8, 6))

# 折线图绘制准确率，增加线条宽度
line1, = ax1.plot(classifiers, accuracies, marker='o', linestyle='-', color=deep_blue, label='Accuracy', linewidth=2)
ax1.set_xlabel('Number of Transformer', fontsize=24)
ax1.set_ylabel('Accuracy (%)', fontsize=24, color=deep_blue)
ax1.tick_params(axis='both', labelsize=20)
ax1.tick_params(axis='y', labelsize=20, colors=deep_blue)
ax1.set_ylim(0, 100)
ax1.set_xticks(np.arange(0, 13, 2))  # 设置X轴刻度为0, 2, 4, ..., 12

# 添加网格线（仅限主轴）
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 使用双轴显示能耗比例，增加线条宽度
ax2 = ax1.twinx()
line2, = ax2.plot(classifiers, [r * 100 for r in energy_ratios], marker='^', linestyle='-', color=deep_red, label='Energy Ratio', linewidth=2)
ax2.set_ylabel('Energy Ratio (%)', fontsize=24, color=deep_red)
ax2.tick_params(axis='y', labelcolor=deep_red, labelsize=20)
ax2.set_ylim(0, 100)

# 设置图例
fig.legend(handles=[line1, line2], labels=['Accuracy', 'Energy Ratio'], loc='upper left', fontsize=20, bbox_to_anchor=(0.1, 0.9))
plt.savefig('motiv_1.png', dpi=300, bbox_inches='tight')
plt.show()
