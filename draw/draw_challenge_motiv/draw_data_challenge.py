# -*- coding: utf-8 -*-
"""
@File    :   draw_data_challenge
@Time    :   2025/5/21 16:32
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   对Challenge中的数据异构性进行画图
"""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate data for five different Non-IID distributions
    x = np.linspace(-10, 10, 1000)

    # Non-IID Distribution 1: Normal Distribution
    normal_dist = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    # Non-IID Distribution 2: Gaussian with different variances
    gaussian_var2 = np.exp(-x ** 2 / 0.2) / np.sqrt(2 * np.pi * 0.2) + np.exp(-(x - 5) ** 2 / 1) / np.sqrt(
        2 * np.pi * 1)

    # Non-IID Distribution 3: Gaussian with different variances
    gaussian_var3 = np.exp(-x ** 2 / 0.3) / np.sqrt(2 * np.pi * 0.3) + np.exp(-(x + 5) ** 2 / 1.5) / np.sqrt(
        2 * np.pi * 1.5)

    # Non-IID Distribution 4: Gaussian with different variances
    gaussian_var4 = np.exp(-x ** 2 / 0.4) / np.sqrt(2 * np.pi * 0.4) + np.exp(-(x - 3) ** 2 / 2) / np.sqrt(
        2 * np.pi * 2)

    # Non-IID Distribution 5: Gaussian with different variances
    gaussian_var5 = np.exp(-x ** 2 / 0.5) / np.sqrt(2 * np.pi * 0.5) + np.exp(-(x - 5) ** 2 / 2) / np.sqrt(
        2 * np.pi * 2)

    # Define the color
    color = (255 / 255, 12 / 255, 235 / 255)

    # Directory to save the figures
    save_dir = r""

    # Plot each distribution in a separate figure and save as SVG

    # Non-IID Distribution 1
    plt.figure(figsize=(5, 5))
    plt.plot(x, normal_dist, label='Normal Distribution', linewidth=6, color=color)
    plt.gca().set_xticklabels([])  # Remove x-axis numbers
    plt.gca().set_yticklabels([])  # Remove y-axis numbers
    # plt.savefig(save_dir + "normal_distribution.svg", format='svg')
    plt.show()
    plt.close()

    # Non-IID Distribution 2
    plt.figure(figsize=(5, 5))
    plt.plot(x, gaussian_var2, label='Gaussian with Different Variances', linewidth=6, color=color)
    plt.gca().set_xticklabels([])  # Remove x-axis numbers
    plt.gca().set_yticklabels([])  # Remove y-axis numbers
    # plt.savefig(save_dir + "gaussian_var2.svg", format='svg')
    plt.show()
    plt.close()

    # Non-IID Distribution 3
    plt.figure(figsize=(5, 5))
    plt.plot(x, gaussian_var3, label='Gaussian with Different Variances', linewidth=6, color=color)
    plt.gca().set_xticklabels([])  # Remove x-axis numbers
    plt.gca().set_yticklabels([])  # Remove y-axis numbers
    # plt.savefig(save_dir + "gaussian_var3.svg", format='svg')
    plt.show()
    plt.close()

    # Non-IID Distribution 4
    plt.figure(figsize=(5, 5))
    plt.plot(x, gaussian_var4, label='Gaussian with Different Variances', linewidth=6, color=color)
    plt.gca().set_xticklabels([])  # Remove x-axis numbers
    plt.gca().set_yticklabels([])  # Remove y-axis numbers
    # plt.savefig(save_dir + "gaussian_var4.svg", format='svg')
    plt.show()
    plt.close()

    # Non-IID Distribution 5
    plt.figure(figsize=(5, 5))
    plt.plot(x, gaussian_var5, label='Gaussian with Different Variances', linewidth=6, color=color)
    plt.gca().set_xticklabels([])  # Remove x-axis numbers
    plt.gca().set_yticklabels([])  # Remove y-axis numbers
    # plt.savefig(save_dir + "gaussian_var5.svg", format='svg')
    plt.show()
    plt.close()
