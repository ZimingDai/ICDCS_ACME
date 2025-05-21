# -*- coding: utf-8 -*-
"""
@File    :   stanford_baseline_header_acc
@Time    :   2025/5/21 16:52
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   Stanford Car上的不同Header比较
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":

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
        "Ours": [0.92187500, 0.90625000, 0.85273537, 0.75568182, 0.59923664, 0.41889313],
        "CNN-Project-EE": [0.6705633627658252, 0.6243004601417734, 0.5527919413008332, 0.40504912324337766,
                           0.10695187165775401, 0.09289889317249099],
        "Linear-EE": [0.8071135430916553, 0.7965427185673424, 0.7558761348091033, 0.6223106578783734,
                      0.3917423206068897,
                      0.21514736973013307],
        "Mixer-EE": [0.7455540355677155, 0.7235418480288521, 0.6195746797661983, 0.46785225718194257,
                     0.19786096256684493,
                     0.12908842183807984],
        "MLP-EE": [0.7703022012187539, 0.7757741574431041, 0.7433155080213903, 0.5642333043153837, 0.2324337768934212,
                   0.18293744559134437],
    }
    linestyle_list = list(line_styles.values())
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    # line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    # colors = ['#d04431', '#49787b', '#7398ba', '#634a5e', '#c86e22']

    colors = ['#344964', '#53a9ca', '#5e997e', '#707b8c', '#475ea4']

    # 创建加粗斜体的字体属性
    bold_italic = FontProperties(weight='bold', style='italic', size=26)

    # 画图
    plt.figure(figsize=(8, 6))

    for i, column in enumerate(df.columns[1:]):
        plt.plot(df["Layer Number"], df[column], color=colors[i], marker=markers[i], markersize=10, label=column,
                 linestyle=linestyle_list[i], linewidth=3)

    # plt.xlabel("Approx. Parameter Count(M)", fontsize=24)
    # plt.ylabel("Accuracy", fontsize=24)
    plt.xticks([0.17, 0.33, 0.5, 0.67, 0.83, 1.0], labels=['14.3', '28.5', '42.7', '56.8', '71.0', '85.2'])
    plt.tick_params(axis='both', labelsize=26)

    # 创建图例
    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(handles, labels, fontsize=26)

    # 设置"Ours"项的字体属性为加粗斜体，并设置字体大小
    for text in legend.get_texts():
        if text.get_text() == "Ours":
            text.set_fontproperties(bold_italic)

    plt.grid(True)
    plt.savefig('figs/stan_header_acc.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
