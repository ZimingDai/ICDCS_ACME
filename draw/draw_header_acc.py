from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Arial'
line_styles = OrderedDict(
    [('solid', (0, ())),
     ('densely dotted', (0, (1, 1))),
     ('densely dashed', (0, (5, 1))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('loosely dotted', (0, (1, 10))),
     ('dotted', (0, (1, 5))),

     ('loosely dashed', (0, (5, 10))),
     ('dashed', (0, (5, 5))),

     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('dashdotted', (0, (3, 5, 1, 5))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
     ])
# 数据
data = {
    "Layer Number": [1.0, 0.83, 0.67, 0.5, 0.33, 0.17],
    # "Ours": [0.9324, 0.9046, 0.8984, 0.8737, 0.8563, 0.8431],
    "Ours": [0.9324, 0.9046, 0.8984, 0.8737, 0.77343750, 0.6731],
    # "CNN-Ignore-EE": [0.8907, 0.8776, 0.8408, 0.7994, 0.664, 0.5507],
    # "CNN-Add-EE": [0.8923, 0.8778, 0.8232, 0.7714, 0.6494, 0.5645],
    "CNN-Project-EE": [0.8991, 0.884, 0.8323, 0.7935, 0.6545, 0.557],
    "Linear-EE": [0.9054, 0.8947, 0.8681, 0.824, 0.7386, 0.6047],
    "Mixer-EE": [0.9012, 0.8892, 0.8691, 0.8073, 0.6605, 0.5687],
    "MLP-EE": [0.9038, 0.8933, 0.8733, 0.8202, 0.7082, 0.6014],
    # "ResMLP-EE": [0.9044, 0.8923, 0.8712, 0.8141, 0.69, 0.5674],
    # "ViT-EE": [0.8972, 0.9015, 0.8703, 0.8178, 0.6856, 0.5774]
}
linestyle_list = list(line_styles.values())
# 转换为 DataFrame
df = pd.DataFrame(data)
# line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
# colors = ['#d04431', '#49787b', '#7398ba', '#634a5e', '#c86e22']

colors = ['#344964', '#53a9ca', '#5e997e', '#707b8c', '#475ea4']

# 创建加粗斜体的字体属性
bold_italic = FontProperties(weight='bold', style='italic', size=20)

# 画图
plt.figure(figsize=(8, 6))

for i, column in enumerate(df.columns[1:]):
    plt.plot(df["Layer Number"], df[column], color=colors[i], marker=markers[i], markersize=10, label=column,
             linestyle=linestyle_list[i], linewidth=3)

plt.xlabel("Approx. Parameter Count(M)", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.xticks([0.17, 0.33, 0.5, 0.67, 0.83, 1.0], labels=['14.3', '28.5', '42.7', '56.8', '71.0', '85.2'])
plt.tick_params(axis='both', labelsize=20)

# 创建图例
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, labels, fontsize=20)

# 设置"Ours"项的字体属性为加粗斜体，并设置字体大小
for text in legend.get_texts():
    if text.get_text() == "Ours":
        text.set_fontproperties(bold_italic)

plt.grid(True)
plt.savefig('header_acc.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
