import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建数据集
data = {
    "Name": ["Efficient-ViT", "Ours(5M)", "MobileViT", "Twins-SVT", "DeViT", "DeDeiTs", "DeCCTs", "Ours(Best)"],
    "Parameters": [5.0, 5.2, 17.43, 24, 23, 23.9, 18.3, 20.4],
    "Accuracy": [0.8108, 0.8515, 0.6738, 0.8672, 0.8798, 0.8934, 0.8743, 0.9234]
}

df = pd.DataFrame(data)
hatch_patterns = ['/', '-']
# 设置柱状图的宽度和位置
bar_width = 0.4
index = np.arange(len(df['Name']))

# 创建图形对象
fig, ax1 = plt.subplots(figsize=(8, 6))

bars1 = ax1.bar(index - bar_width / 2, df['Parameters'], bar_width, label='Parameters', color='#6caa8a',
                hatch=hatch_patterns[0], edgecolor='white')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制准确率的柱状图
bars2 = ax2.bar(index + bar_width / 2, df['Accuracy'], bar_width, label='Accuracy', color='#3a5171',
                hatch=hatch_patterns[1], edgecolor='white')

# 设置准确率的范围
ax2.set_ylim(0.65, 0.95)

# 设置网格
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置标签和标题
ax1.set_ylabel('Parameter Count(M)', fontsize=24)
ax2.set_ylabel('Accuracy', fontsize=24)
ax1.set_xticks(index)  # 删除横坐标刻度
ax1.set_xticklabels(df['Name'], rotation=25, ha='right', fontsize=20)

# 设置纵坐标刻度字体大小
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

# 设置准确率的y_ticks为小数点后一位
accuracy_ticks = np.round(np.arange(0.65, 1.0, 0.05), 1)
ax2.set_yticks(accuracy_ticks)

# 设置横坐标颜色
colors = ['black'] * len(df['Name'])
colors[1] = 'red'
colors[-1] = 'red'
for ticklabel, tickcolor in zip(ax1.get_xticklabels(), colors):
    ticklabel.set_color(tickcolor)

# 显示图例
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=20, framealpha=0.4)

# 布局调整
plt.tight_layout()
plt.savefig('baseline.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
