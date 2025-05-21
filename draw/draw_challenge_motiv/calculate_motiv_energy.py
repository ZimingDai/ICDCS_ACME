# -*- coding: utf-8 -*-
"""
@File    :   calculate_motiv_energy
@Time    :   2025/5/21 16:26
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   计算motivation中不同深度的能耗值
"""

if __name__ == "__main__":
    # 使用假设值计算能耗
    G_n = 1
    G_n_b = 1  # 假设 G_n^b 与 G_n 相等
    hat_G_n = 0.1
    I_n = 0.5
    hat_I_n = 0.05
    p_n = 0.2
    b = 0.5
    w_n = 1


    # 定义能耗计算函数
    def calculate_energy_ratio(d, G_n, hat_G_n, w_n, p_n, G_n_b, b, P_full):
        # 计算对于给定d的能耗
        P_d = d * (G_n + hat_G_n * w_n * d) + d * p_n * G_n_b * b
        # 计算能耗比例
        ratio = P_d / P_full
        return ratio


    # 满12层的能耗
    P_full = 12 * (G_n + hat_G_n * w_n * 12) + 12 * p_n * G_n_b * b

    # 对每个d进行计算
    ratios = [calculate_energy_ratio(d, G_n, hat_G_n, w_n, p_n, G_n_b, b, P_full) for d in range(1, 13)]

    # 打印每个d的能耗比例
    for d, ratio in enumerate(ratios, 1):
        print(f"能耗比例 P({d})/P(12): {ratio:.2f}")
