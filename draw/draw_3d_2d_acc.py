import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rcParams['font.family'] = 'Arial'

# 数据
cnnignore_x075 = [0.6762, 0.7883, 0.8434, 0.857]
linear_x075 = [0.6206, 0.8075, 0.8723, 0.8955]
our_x075 = [0.8203, 0.9062, 0.9268, 0.9365]

cnnignore_y075 = [0.6001, 0.7596, 0.8434, 0.8785]
linear_y075 = [0.399, 0.7337, 0.8723, 0.906]
our_y075 = [0.8186, 0.8671, 0.9268, 0.9367]

# 颜色设置
colors = {
    "Ours": [76 / 255, 167 / 255, 164 / 255],
    "CNN": [128 / 255, 213 / 255, 130 / 255],
    "Linear": [105 / 255, 49 / 255, 119 / 255]
}
line_styles = ['-', '--', '-.']
# 图1 - 数据 x=0.75
plt.figure(figsize=(7, 4))
plt.plot(our_x075, label='Ours', color=colors['Ours'], linewidth=5, marker='o', markersize=10, linestyle=line_styles[0])
plt.plot(cnnignore_x075, label='CNN', color=colors['CNN'], linewidth=5, marker='D', markersize=10,
         linestyle=line_styles[1])
plt.plot(linear_x075, label='Linear', color=colors['Linear'], linewidth=5, marker='s', markersize=10,
         linestyle=line_styles[2])

plt.xticks([0, 1, 2, 3], [0.25, 0.5, 0.75, 1.0], fontsize=24)

plt.xlabel('Width', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig('plot1_x075.svg', dpi=300, bbox_inches='tight')
plt.show()

# 图2 - 数据 y=0.75
plt.figure(figsize=(7, 4))
plt.plot(our_y075, label='Ours', color=colors['Ours'], linewidth=5, marker='o', markersize=10, linestyle=line_styles[
    0])
plt.plot(cnnignore_y075, label='CNN', color=colors['CNN'], linewidth=5, marker='D', markersize=10, linestyle=
line_styles[1])
plt.plot(linear_y075, label='Linear', color=colors['Linear'], linewidth=5, marker='s', markersize=10, linestyle=
line_styles[2])

# 设置横坐标
plt.xticks([0, 1, 2, 3], [0.25, 0.5, 0.75, 1.0], fontsize=24)

plt.xlabel('Depth', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig('plot2_y075.svg', dpi=300, bbox_inches='tight')
plt.show()
