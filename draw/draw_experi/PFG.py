# -*- coding: utf-8 -*-
"""
@File    :   PFG
@Time    :   2025/5/21 16:49
@Author  :   phoenixdai
@Version :   1.0
@Site    :   http://phoenixdai.cn
@Desc    :   PFG效果图
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

if __name__ == "__main__":

    # Updated data based on the new input
    data_latest = {
        "Model": ["Optimal", "Highest Accuracy", "Largest Size", "Random"],
        "Predicted Accuracy": [0.895977, 0.915464, 0.913366, 0.753609],
        "Model Size (params)": [0.666729, 0.750042, 0.750063, 0.74375],
        "Total Energy (Ws)": [0.568546, 0.649021, 0.716418, 0.68180],
        "Time": [0.000475, 0.001650, 0.001558, 0.000583]
    }

    df_latest = pd.DataFrame(data_latest)

    # Calculate Model Efficiency Ratio (MER)
    df_latest["MER"] = df_latest["Predicted Accuracy"] / df_latest["Model Size (params)"]

    # Calculate Energy Efficiency Ratio (EER)
    df_latest["EER"] = df_latest["Predicted Accuracy"] / df_latest["Total Energy (Ws)"]

    # Calculate the trade-off score with the updated data
    df_latest["Trade-off Score"] = df_latest["Predicted Accuracy"] - 0.5 * (
            df_latest["Model Size (params)"] + df_latest["Total Energy (Ws)"])

    print(df_latest["Trade-off Score"])
    # Define colors for the models using the provided color codes
    colors_filtered = ['#f1b069', '#6490b3', '#7cb89f', '#33745c']

    # Define different hatch patterns for the bars
    hatch_patterns = ['/', '\\', '|', '-']

    # Set font properties
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['ytick.labelsize'] = 30

    # Plot Predicted Accuracy
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["Predicted Accuracy"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])

    ax.set_xticklabels([])
    ax.set_ylim(0.4, df_latest["Predicted Accuracy"].max() + 0.15)
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    # plt.savefig('predicted_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Model Size
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["Model Size (params)"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_ylim(0.4, df_latest["Model Size (params)"].max() + 0.1)
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    # plt.savefig('model_size.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Total Energy
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["Total Energy (Ws)"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_ylim(0.4, df_latest["Total Energy (Ws)"].max() + 0.1)
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('total_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Time
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["Time"].iloc[idx] * 1000, color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    plt.tight_layout()
    # plt.savefig('time.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot EER
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["EER"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_ylim(0.9, df_latest["EER"].max() + 0.04)
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    # plt.savefig('eer.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot MER
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["MER"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_ylim(0.9, df_latest["MER"].max() + 0.04)
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    # plt.savefig('mer.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Trade-off Score
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, model in enumerate(df_latest["Model"]):
        bar = ax.bar(model, df_latest["Trade-off Score"].iloc[idx], color=colors_filtered[idx])
        bar[0].set_hatch(hatch_patterns[idx])
        bar[0].set_edgecolor('black')
    ax.set_xticklabels([])
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.tight_layout()
    # plt.savefig('trade_off_score.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
