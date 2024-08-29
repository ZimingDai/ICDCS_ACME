#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   eval_PFG.py
@Time    :   2024/06/21 15:37:54
@Author  :   PhoenixDai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   对PFG进行实验
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
import warnings
import logging

# 禁用警告
warnings.filterwarnings("ignore")

# 设置日志记录
logging.basicConfig(filename='../log/PFG_part/model_evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
np.random.seed(0)

# 设置浮点数显示格式
pd.options.display.float_format = '{:.6f}'.format


def calculate_model_size(width_ratios, depth_ratios, heads_per_head=768, params_per_layer=3060000, full_width=12,
                         full_depth=12):
    results = []
    for w_ratio in width_ratios:
        for d_ratio in depth_ratios:
            w_n_B = int(full_width * w_ratio)
            d_n_B = int(full_depth * d_ratio)
            size = (w_n_B * heads_per_head) + (d_n_B * params_per_layer)
            results.append({
                "Width Ratio": w_ratio,
                "Depth Ratio": d_ratio,
                "Width": w_n_B,
                "Depth": d_n_B,
                "Model Size (params)": size
            })
    results_df = pd.DataFrame(results)
    return results_df


def calculate_energy_consumption(width_ratios, depth_ratios, G_n=100, delta_G_n=10, L_n=1, delta_L_n=0.1, p_n=128,
                                 G_n_beta=50, k=10, full_width=12, full_depth=12):
    results = []
    for w_ratio in width_ratios:
        for d_ratio in depth_ratios:
            w_n_B = int(full_width * w_ratio)
            d_n_B = int(full_depth * d_ratio)
            P_n = (G_n + delta_G_n * w_n_B * d_n_B) + p_n * G_n_beta
            T_n = (L_n + delta_L_n * w_n_B * d_n_B)
            E_epoch = P_n * T_n  # 单个 epoch 的能耗 (瓦秒)
            E_total = k * E_epoch  # 总能耗 (瓦秒)
            results.append({
                "Width Ratio": w_ratio,
                "Depth Ratio": d_ratio,
                "Width": w_n_B,
                "Depth": d_n_B,
                "Power (W)": P_n,
                "Latency (s)": T_n,
                "Energy per Epoch (Ws)": E_epoch,
                "Total Energy (Ws)": E_total
            })
    results_df = pd.DataFrame(results)
    return results_df


def fit_and_predict_accuracy(existing_data, new_width_ratios, new_depth_ratios, degree=2):
    df = pd.DataFrame(existing_data, columns=['Depth Ratio', 'Width Ratio', 'Accuracy'])
    X = df[['Depth Ratio', 'Width Ratio']]
    y = df['Accuracy']

    # 使用多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(X_poly, y)

    # 生成新的预测数据
    new_ratios = np.array([(d, w) for d in new_depth_ratios for w in new_width_ratios])
    new_X_poly = poly.transform(new_ratios)
    new_accuracies = model.predict(new_X_poly)

    # 确保预测准确率在0到1之间
    new_accuracies = np.clip(new_accuracies, 0, 1)

    results = []
    for i, (d_ratio, w_ratio) in enumerate(new_ratios):
        results.append({
            "Depth Ratio": d_ratio,
            "Width Ratio": w_ratio,
            "Predicted Accuracy": new_accuracies[i]
        })

    results_df = pd.DataFrame(results)
    return results_df


def normalize(df, columns):
    result = df.copy()
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        result[column] = (df[column] - min_val) / (max_val - min_val)
    return result


def calculate_grid_coordinates(row, ideal_point, non_ideal_point, sigma, K):
    r = (ideal_point - non_ideal_point + 2 * sigma) / K
    coordinates = np.ceil((ideal_point - row + sigma) / r)
    return coordinates


def multi_objective_optimization(combined_df, size_constraint, gamma_p=0.1, K=10, sigma=1e-6):
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 设置理想点和非理想点
    ideal_point = np.array([1, 0, 0])
    non_ideal_point = np.array([0, 1, 1])

    combined_df[['accuracy_grid', 'delay_grid', 'energy_grid']] = combined_df.apply(
        lambda row: calculate_grid_coordinates(row[['Predicted Accuracy', 'Latency (s)', 'Total Energy (Ws)']].values,
                                               ideal_point, non_ideal_point, sigma, K),
        axis=1,
        result_type='expand'
    )

    # 求解子问题，找到每个网格的最优值
    pareto_front = []
    for l in range(1, K + 1):
        for m in range(1, K + 1):
            for n in range(1, K + 1):
                sub_df = combined_df[(combined_df['accuracy_grid'] == l) & (combined_df['delay_grid'] == m) & (
                        combined_df['energy_grid'] == n)]
                if not sub_df.empty:
                    # 计算每个点到理想点的距离
                    sub_df['distance_to_ideal'] = sub_df[
                        ['Predicted Accuracy', 'Latency (s)', 'Total Energy (Ws)']].apply(
                        lambda x: np.linalg.norm(x - ideal_point), axis=1)
                    # 选择距离理想点最近的点
                    pareto_optimal_idx = sub_df['distance_to_ideal'].idxmin()
                    pareto_optimal = sub_df.loc[pareto_optimal_idx]
                    pareto_front.append(pareto_optimal)

    pareto_front_df = pd.DataFrame(pareto_front).drop_duplicates()

    # 选择最优模型
    # 使用 size_constraint 进行筛选
    start_time = time.time()
    optimal_model = \
        pareto_front_df[pareto_front_df['Model Size (params)'] <= size_constraint].sort_values(by='Predicted Accuracy',
                                                                                               ascending=False).iloc[0]
    pareto_time = time.time() - start_time

    # 只保留所需的字段
    optimal_model = optimal_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    optimal_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    optimal_model['pareto_time'] = pareto_time

    return optimal_model


def random_selection(combined_df, size_constraint):
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 筛选满足大小限制的模型
    valid_models = combined_df[combined_df['Model Size (params)'] <= size_constraint]

    start_time = time.time()

    # 随机选择一个模型
    random_model = valid_models.sample(n=1, random_state=0).iloc[0]

    # 记录时间
    random_time = time.time() - start_time

    # 只保留所需的字段
    random_model = random_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    random_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    random_model['random_time'] = random_time

    return random_model


def highest_accuracy_selection(combined_df, size_constraint):
    start_time = time.time()
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 筛选满足大小限制的模型
    valid_models = combined_df[combined_df['Model Size (params)'] <= size_constraint]

    # 选择准确率最高的模型
    highest_accuracy_model = valid_models.sort_values(by='Predicted Accuracy', ascending=False).iloc[0]

    # 记录时间
    highest_accuracy_time = time.time() - start_time

    # 只保留所需的字段
    highest_accuracy_model = highest_accuracy_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    highest_accuracy_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    highest_accuracy_model['highest_accuracy_time'] = highest_accuracy_time

    return highest_accuracy_model


def largest_size_selection(combined_df, size_constraint):
    start_time = time.time()
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 筛选满足大小限制的模型
    valid_models = combined_df[combined_df['Model Size (params)'] <= size_constraint]

    # 选择大小最大的模型
    largest_size_model = valid_models.sort_values(by='Model Size (params)', ascending=False).iloc[0]

    # 记录时间
    largest_size_time = time.time() - start_time

    # 只保留所需的字段
    largest_size_model = largest_size_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    largest_size_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    largest_size_model['largest_size_time'] = largest_size_time

    return largest_size_model


def traverse_selection(combined_df, size_constraint):
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 筛选满足大小限制的模型
    valid_models = combined_df[combined_df['Model Size (params)'] <= size_constraint]

    # 遍历选择 (能耗 + 大小 - 准确率) 最低的模型
    start_time = time.time()
    best_model = None
    min_objective_value = float('inf')

    for idx, row in valid_models.iterrows():
        objective_value = 1 * (row['Total Energy (Ws)'] + row['Model Size (params)']) - row['Predicted Accuracy']
        if objective_value < min_objective_value:
            min_objective_value = objective_value
            best_model = row

    traverse_time = time.time() - start_time

    # 只保留所需的字段
    best_model = best_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    best_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    best_model['traverse_time'] = traverse_time

    return best_model


def worst_random_model_selection(combined_df, size_constraint):
    start_time = time.time()
    # 归一化能耗和模型大小
    combined_df = normalize(combined_df, ['Latency (s)', 'Total Energy (Ws)', 'Model Size (params)'])

    # 筛选满足大小限制且准确率不为0的模型
    valid_models = combined_df[
        (combined_df['Model Size (params)'] <= size_constraint) & (combined_df['Predicted Accuracy'] > 0)]

    # 选择准确率最低、能耗最高、模型大小最大的模型
    worst_model = valid_models.sort_values(by=['Predicted Accuracy', 'Total Energy (Ws)', 'Model Size (params)'],
                                           ascending=[True, False, False]).iloc[0]

    # 记录时间
    worst_model_time = time.time() - start_time

    # 只保留所需的字段
    worst_model = worst_model[
        ['Predicted Accuracy', 'Model Size (params)', 'Total Energy (Ws)', 'Width Ratio', 'Depth Ratio']]
    worst_model.columns = ['Accuracy', 'Size', 'Energy Consumption', 'Width Ratio', 'Depth Ratio']
    worst_model['worst_model_time'] = worst_model_time

    return worst_model


if __name__ == "__main__":
    width_ratios = np.linspace(0, 1, 20)
    depth_ratios = np.linspace(0, 1, 20)

    # 原始准确率数据
    existing_data = [
        (1.0, 1.0, 0.9207),
        (1.0, 0.75, 0.906),
        (1.0, 0.5, 0.8677),
        (1.0, 0.25, 0.7054),
        (0.75, 1.0, 0.8955),
        (0.75, 0.75, 0.8723),
        (0.75, 0.5, 0.8075),
        (0.75, 0.25, 0.6206),
        (0.5, 1.0, 0.7921),
        (0.5, 0.75, 0.7337),
        (0.5, 0.5, 0.6401),
        (0.5, 0.25, 0.4236),
        (0.25, 1.0, 0.4381),
        (0.25, 0.75, 0.399),
        (0.25, 0.5, 0.3375),
        (0.25, 0.25, 0.2262),
    ]

    # 计算能耗
    energy_consumption_df = calculate_energy_consumption(width_ratios, depth_ratios)
    # 计算模型大小
    model_size_df = calculate_model_size(width_ratios, depth_ratios)
    # 预测准确率
    predicted_accuracy_df = fit_and_predict_accuracy(existing_data, width_ratios, depth_ratios, degree=2)

    # 合并数据
    combined_df = energy_consumption_df.merge(model_size_df, on=["Width Ratio", "Depth Ratio"])
    combined_df = combined_df.merge(predicted_accuracy_df, on=["Width Ratio", "Depth Ratio"])

    # 保存到文件
    combined_df.to_csv("../log/PFG_part/model_results.csv", index=False)

    # 设置模型大小限制
    size_constraint = 0.8

    # 多目标优化
    optimal_model = multi_objective_optimization(combined_df, size_constraint)
    logging.info("Optimal Model (Multi-objective Optimization):")
    logging.info(optimal_model)

    # 随机选择模型
    random_model = random_selection(combined_df, size_constraint)
    logging.info("Randomly Selected Model:")
    logging.info(random_model)

    # 选择准确率最高的模型
    highest_accuracy_model = highest_accuracy_selection(combined_df, size_constraint)
    logging.info("Highest Accuracy Model:")
    logging.info(highest_accuracy_model)

    # 选择大小最大的模型
    largest_size_model = largest_size_selection(combined_df, size_constraint)
    logging.info("Largest Size Model:")
    logging.info(largest_size_model)

    # 遍历选择能耗+大小-准确率最低的模型
    traverse_model = traverse_selection(combined_df, size_constraint)
    logging.info("Traverse Selected Model:")
    logging.info(traverse_model)

    # 遍历选择能耗+大小-准确率最低的模型
    worst_model = worst_random_model_selection(combined_df, size_constraint)
    logging.info("Worst Selected Model:")
    logging.info(worst_model)
