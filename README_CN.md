
# ACME: 基于分布式系统的大模型自适应定制方法

[![切换为英文版](https://img.shields.io/badge/GitHub-English-blue?logo=github)](./README.md)

本仓库包含 ICDCS 2025 接收论文 **《ACME: Adaptive Customization of Large Models via Distributed Systems》** 的代码。

## 📚简介

ACME（Adaptive Customization of Large Models via Distributed Systems）是一个用于在分布式系统中对大模型进行自适应定制的系统框架。面对传统大模型在设备端部署时存在的**模型不匹配、资源受限、数据异构性强**等挑战，ACME 提出一种**双向单环分布式定制系统**，通过**“从云到边再到端”的协同机制**，逐步完成模型骨干与头部的个性化生成。系统在**不上传本地数据的前提下**，实现了高准确率与低通信开销的平衡，显著提升了模型的部署效率与端侧适配性。

## 📁 项目结构

```
├── data/                    # 数据集目录（请放置 CIFAR-100 或 Stanford Cars）
├── draw/                    # 绘图脚本与输出
│		├── draw_challenge_motiv/            
│   └── draw_experi/ 
├── log/                     # 实验运行日志
├── model/                   # 模型定义与预训练模型文件
├── moti/                    # 动机分析相关内容
│   └── origin_motiv/
│       └── log/             # 原始动机实验日志
├── runs/                    # 实验结果与中间文件
└── src/                     # 项目主代码
    ├── backup/              # 替换 Transformers 库的自定义代码
    ├── shell/               # 自动化执行的 Shell 脚本
    └── third_part/					 # ⚠️论文第三部分细粒度个性化的独立工程⚠️
```

## 🛠️ 环境准备

1. 安装依赖包：

   ```bash
   pip install -r requirements.txt
   ```

2. 下载预训练模型并放置在 `model/` 目录中：

   - [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

3. 下载以下数据集并放置在 `data/` 目录中：

   - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
   - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

4. ⚠️ **重要操作：替换 Transformers 库源码文件**

   为了实现自定义行为，请务必执行以下操作替换 Transformers 的部分源码：

   - 定位你本地环境中 Transformers 库的安装路径（通常为 `site-packages/transformers/`）
   - 将 `src/backup/` 中的文件 **手动复制并覆盖** 到上述目录中对应的位置

   > ⚠ 警告：此操作将修改 Transformers 的默认实现，请确保你了解替换内容，并建议在虚拟环境中操作以避免影响其他项目。

5. ⚠️ **注意：`src/third_part/` 是论文第三部分的独立工程**

   


## 🚀 实验运行

1. 云端预训练：

   ```bash
   python cloud_pretrain.py
   ```

2. 运行 DynaViTw 定制脚本：

   ```bash
   bash ./src/shell/run_cloud1.sh
   ```

3. 运行 DynaViT 定制脚本：

   ```bash
   bash ./src/shell/run_cloud2.sh
   ```

4. 运行边缘侧 NAS：

   ```bash
   bash ./src/shell/run_nas.sh
   ```

5. ⚠️ 若要运行第三部分细粒度个性化实验，请进入 `src/third_part/`，参考“环境准备”部分说明，单独运行：

    ```bash
    cd src/third_part
    bash run.sh
    ```

## 📦 模型与数据集链接

| 类型 | 名称 | 链接 |
|------|------|------|
| 模型 | vit-base-patch16-224 | [Hugging Face 下载](https://huggingface.co/google/vit-base-patch16-224) |
| 数据集 | CIFAR-100 | [CIFAR 官网](https://www.cs.toronto.edu/~kriz/cifar.html) |
| 数据集 | Stanford Cars | [Stanford 官网](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) |
