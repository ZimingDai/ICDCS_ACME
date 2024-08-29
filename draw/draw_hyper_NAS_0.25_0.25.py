import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 创建一个以元组为键，数值为值的字典
data_dict = {
    ('B=1', 'U=1'): 53.664444,
    ('B=1', 'U=2'): 55.166331,
    ('B=1', 'U=3'): 55.468750,
    ('B=2', 'U=1'): 55.480000,
    ('B=2', 'U=2'): 57.812500,
    ('B=2', 'U=3'): 59.375000,
    ('B=3', 'U=1'): 56.250000,
    ('B=3', 'U=2'): 60.156250,
    ('B=3', 'U=3'): 62.934603
}

# 将数据乘以0.01
data_dict = {key: value * 0.01 for key, value in data_dict.items()}

# 提取 Cell 和 Layer 信息
cells = sorted(set(k[0] for k in data_dict.keys()))
layers = sorted(set(k[1] for k in data_dict.keys()))

# 构造 DataFrame
df = pd.DataFrame(index=cells, columns=layers)
for (cell, layer), value in data_dict.items():
    df.at[cell, layer] = value

# 绘制热力图，并调整数字注释和图例字体大小
plt.figure(figsize=(7, 3))
ax = sns.heatmap(df.astype(float), annot=True, cmap='Blues', fmt=".3f",
                 annot_kws={'size': 20}, vmin=0.53, vmax=0.63)  # 设置图例范围
colorbar = ax.collections[0].colorbar

# 设置 colorbar 的标签
ticks = np.arange(0.53, 0.63 + 0.01, 0.03)
colorbar.set_ticks(ticks)
colorbar.set_ticklabels([f"{x:.2f}" for x in ticks])  # 设置图例标签为两位小数
colorbar.ax.tick_params(labelsize=20)  # 设置图例字体大小

ax.tick_params(axis='both', which='major', labelsize=20)  # 设置刻度标签的字体大小为20

plt.savefig('2_hyper.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
