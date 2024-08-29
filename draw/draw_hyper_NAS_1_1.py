import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 创建一个以元组为键，数值为值的字典
data_dict = {
    ('B=1', 'U=1'): 0.9453125,
    ('B=1', 'U=2'): 0.94131111,
    ('B=1', 'U=3'): 0.93988889,
    ('B=2', 'U=1'): 0.93904444,
    ('B=2', 'U=2'): 0.93522222,
    ('B=2', 'U=3'): 0.93433333,
    ('B=3', 'U=1'): 0.93568889,
    ('B=3', 'U=2'): 0.93417730,
    ('B=3', 'U=3'): 0.93366276
}

# 提取 Cell 和 Layer 信息
cells = sorted(set(k[0] for k in data_dict.keys()))
layers = sorted(set(k[1] for k in data_dict.keys()))

# 构造 DataFrame
df = pd.DataFrame(index=cells, columns=layers)
for (cell, layer), value in data_dict.items():
    df.at[cell, layer] = value

# 绘制热力图，并调整数字注释和图例字体大小
plt.figure(figsize=(7, 3))
ax = sns.heatmap(df.astype(float), annot=True, cmap='Blues', fmt=".4f",
                 annot_kws={'size': 20}, vmin=0.933, vmax=0.945)  # 设置图例范围
colorbar = ax.collections[0].colorbar

# 设置 colorbar 的标签
ticks = np.arange(0.93, 0.94 + 0.01, 0.01)
colorbar.set_ticks(ticks)
colorbar.set_ticklabels([f"{x:.2f}" for x in ticks])  # 设置图例标签为小数
colorbar.ax.tick_params(labelsize=20)  # 设置图例字体大小

ax.tick_params(axis='both', which='major', labelsize=20)  # 设置刻度标签的字体大小为20

plt.savefig('1_hyper.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
