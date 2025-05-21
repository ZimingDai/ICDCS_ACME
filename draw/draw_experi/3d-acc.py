# -*- coding: utf-8 -*-
"""
@File    :   3d-acc
@Time    :   2025/5/21 16:38
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   多Header的3D图
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.interpolate import griddata

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'

    # Define the data:深度、宽度
    cnnignore = {
        (1.0, 1.0): 0.8907,
        (1.0, 0.75): 0.8785,
        (1.0, 0.5): 0.842,
        (1.0, 0.25): 0.7271,
        (0.75, 1.0): 0.857,
        (0.75, 0.75): 0.8434,
        (0.75, 0.5): 0.7883,
        (0.75, 0.25): 0.6762,
        (0.5, 1.0): 0.8009,
        (0.5, 0.75): 0.7596,
        (0.5, 0.5): 0.7127,
        (0.5, 0.25): 0.5303,
        (0.25, 1.0): 0.6234,
        (0.25, 0.75): 0.6001,
        (0.25, 0.5): 0.5767,
        (0.25, 0.25): 0.5082,
    }

    cnnproject = {
        (1.0, 1.0): 0.8991,
        (1.0, 0.75): 0.8852,
        (1.0, 0.5): 0.8581,
        (1.0, 0.25): 0.746,
        (0.75, 1.0): 0.8656,
        (0.75, 0.75): 0.8462,
        (0.75, 0.5): 0.7991,
        (0.75, 0.25): 0.6787,
        (0.5, 1.0): 0.8059,
        (0.5, 0.75): 0.7421,
        (0.5, 0.5): 0.7099,
        (0.5, 0.25): 0.5791,
        (0.25, 1.0): 0.6158,
        (0.25, 0.75): 0.5975,
        (0.25, 0.5): 0.5285,
        (0.25, 0.25): 0.5093,
    }

    linear = {
        (1.0, 1.0): 0.9207,
        (1.0, 0.75): 0.906,
        (1.0, 0.5): 0.8677,
        (1.0, 0.25): 0.7054,
        (0.75, 1.0): 0.8955,
        (0.75, 0.75): 0.8723,
        (0.75, 0.5): 0.8075,
        (0.75, 0.25): 0.6206,
        (0.5, 1.0): 0.7921,
        (0.5, 0.75): 0.7337,
        (0.5, 0.5): 0.6401,
        (0.5, 0.25): 0.4236,
        (0.25, 1.0): 0.4381,
        (0.25, 0.75): 0.399,
        (0.25, 0.5): 0.3375,
        (0.25, 0.25): 0.2262,
    }

    our = {
        (1.0, 1.0): 0.9524,
        (1.0, 0.75): 0.9367,
        (1.0, 0.5): 0.9234,
        (1.0, 0.25): 0.84,
        (0.75, 1.0): 0.9365,
        (0.75, 0.75): 0.9268,
        (0.75, 0.5): 0.90625000,
        (0.75, 0.25): 0.82031250,
        (0.5, 1.0): 0.9085,
        (0.5, 0.75): 0.86718750,
        (0.5, 0.5): 0.85156250,
        (0.5, 0.25): 0.75,
        (0.25, 1.0): 0.8963,
        (0.25, 0.75): 0.8182,
        (0.25, 0.5): 0.77343750,
        (0.25, 0.25): 0.6767,
    }

    # Prepare data for plotting
    widths = [0.25, 0.5, 0.75, 1.0]
    depths = [0.25, 0.5, 0.75, 1.0]

    # Create meshgrid
    X, Y = np.meshgrid(depths, widths)

    # Extract Z values
    Z1 = np.array([cnnignore[(d, w)] for d in depths for w in widths]).reshape(4, 4)
    Z2 = np.array([cnnproject[(d, w)] for d in depths for w in widths]).reshape(4, 4)
    Z3 = np.array([linear[(d, w)] for d in depths for w in widths]).reshape(4, 4)
    Z4 = np.array([our[(d, w)] for d in depths for w in widths]).reshape(4, 4)

    # Create finer meshgrid for smoother surface
    X_fine, Y_fine = np.meshgrid(np.linspace(0.25, 1.0, 50), np.linspace(0.25, 1.0, 50))

    # Interpolate Z values for finer meshgrid
    Z1_fine = griddata((X.flatten(), Y.flatten()), Z1.flatten(), (X_fine, Y_fine), method='cubic')
    Z2_fine = griddata((X.flatten(), Y.flatten()), Z2.flatten(), (X_fine, Y_fine), method='cubic')
    Z3_fine = griddata((X.flatten(), Y.flatten()), Z3.flatten(), (X_fine, Y_fine), method='cubic')
    Z4_fine = griddata((X.flatten(), Y.flatten()), Z4.flatten(), (X_fine, Y_fine), method='cubic')

    # Custom transformation for Z-axis positions to create non-uniform spacing
    z_positions = np.linspace(0.2, 1, 8)
    z_custom_positions = np.array([0, 1, 2, 3.5, 5.5, 8, 12, 18])  # Custom positions


    def custom_zscale(value):
        return np.interp(value, z_positions, z_custom_positions)


    def inverse_custom_zscale(value):
        return np.interp(value, z_custom_positions, z_positions)


    # Apply the custom transformation to Z values
    Z1_transformed = custom_zscale(Z1_fine)
    Z2_transformed = custom_zscale(Z2_fine)
    Z3_transformed = custom_zscale(Z3_fine)
    Z4_transformed = custom_zscale(Z4_fine)

    # Plotting 3D surface plots with custom non-linear scale on the Z-axis
    fig = plt.figure(figsize=(16, 12))  # Increased figure size
    ax = fig.add_subplot(111, projection='3d')

    # # 生成 x 和 y 数据
    # x = np.linspace(0.2, 1, 9)
    # y = np.linspace(0.2, 1, 9)
    # X, Y = np.meshgrid(x, y)
    #
    # # 生成 z 数据
    # z1 = np.linspace(0, 18, 9)
    # Z1 = np.meshgrid(z1, z1)[0]
    #
    # # 绘制平面 x=0.75 的部分
    # ax.plot_surface(X * 0 + 0.75, Y, Z1, color='gray', alpha=0.2)
    #
    # # 绘制平面 y=0.75 的部分
    # Z2 = 22.5 * Y - 4.5
    # ax.plot_surface(X, Y * 0 + 0.75, Z2, color='gray', alpha=0.2)
    #
    # ax.plot([0.75] * len(x), x, Z1[:, 0], color='black', linestyle='--', linewidth=2)  # 左侧边
    # ax.plot([0.75] * len(x), x, Z1[:, -1], color='black', linestyle='--', linewidth=2)  # 右侧边
    # ax.plot([0.75, 0.75], [0.2, 0.2], [0, 18], color='black', linestyle='--', linewidth=2)  # 前侧
    # ax.plot([0.75, 0.75], [1, 1], [0, 18], color='black', linestyle='--', linewidth=2)  # 后侧
    #
    #
    # ax.plot([0.2, 0.2], [0.75, 0.75], [0, 18], color='black', linestyle='--', linewidth=2)  # 前侧
    # ax.plot([1, 1], [0.75, 0.75], [0, 18], color='black', linestyle='--', linewidth=2)  # 后侧
    # ax.plot(x, [0.75] * len(x), Z1[:, 0], color='black', linestyle='--', linewidth=2)  # 左侧边
    # ax.plot(x, [0.75] * len(x), Z1[:, -1], color='black', linestyle='--', linewidth=2)  # 右侧边

    # Apply custom colors (adjust the color codes as needed)
    facecolors1 = np.ones((Z1_transformed.shape[0], Z1_transformed.shape[1], 4))
    facecolors1[:, :, :3] = [128 / 255, 213 / 255, 130 / 255]  # Dark blue color
    facecolors1[:, :, 3] = 0  # Alpha 0.7

    # facecolors2 = np.ones((Z2_transformed.shape[0], Z2_transformed.shape[1], 4))
    # facecolors2[:, :, :3] =  # Dark green color
    # facecolors2[:, :, 3] = 0  # Alpha 0.7

    facecolors3 = np.ones((Z3_transformed.shape[0], Z3_transformed.shape[1], 4))
    facecolors3[:, :, :3] = [105 / 255, 49 / 255, 119 / 255]  # Dark red color
    facecolors3[:, :, 3] = 0  # Alpha 0.7

    facecolors4 = np.ones((Z4_transformed.shape[0], Z4_transformed.shape[1], 4))
    facecolors4[:, :, :3] = [76 / 255, 167 / 255, 164 / 255]  # Dark red color
    facecolors4[:, :, 3] = 0  # Alpha 0.7

    # Plot first group with uniform color
    ax.plot_surface(X_fine, Y_fine, Z1_transformed, facecolors=facecolors1, alpha=0.7)
    # Plot second group with uniform color
    # ax.plot_surface(X_fine, Y_fine, Z2_transformed, facecolors=facecolors2, alpha=0.5)
    # Plot third group with uniform color
    ax.plot_surface(X_fine, Y_fine, Z3_transformed, facecolors=facecolors3, alpha=0.7)

    ax.plot_surface(X_fine, Y_fine, Z4_transformed, facecolors=facecolors4, alpha=0.7)

    ax.view_init(elev=15., azim=-50)
    ax.set_zlim(0, 15)  # Limit the z-axis to show up to 15 units (corresponding to 0.95)

    # Set custom ticks for the Z-axis
    ax.set_zticks(z_custom_positions)
    ax.set_zticklabels([f"{z:.1f}" for z in z_positions])

    # 设置坐标轴标签字体大小为20
    ax.set_xlabel('Depth', fontsize=27, labelpad=17)
    ax.set_ylabel('Width', fontsize=27, labelpad=17)
    ax.set_zlabel('Accuracy', fontsize=27, labelpad=17)

    # 设置坐标轴刻度字体大小为20
    ax.tick_params(axis='both', labelsize=25)

    # 添加图例
    legend_elements = [
        Patch(facecolor=[76 / 255, 167 / 255, 164 / 255], edgecolor=[76 / 255, 167 / 255, 164 / 255], label='Ours'),
        Patch(facecolor=[105 / 255, 49 / 255, 119 / 255], edgecolor=[105 / 255, 49 / 255, 119 / 255], label='Linear'),
        Patch(facecolor=[128 / 255, 213 / 255, 130 / 255], edgecolor=[128 / 255, 213 / 255, 130 / 255], label='CNN'),
    ]

    ax.legend(handles=legend_elements, fontsize=25)

    # plt.savefig('figs/temp.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
