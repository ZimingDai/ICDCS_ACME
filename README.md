
# FormerTailor: Customized Large Model Adaptation in Heterogeneous Cloud-Edge-Device Systems

本仓库包含论文"**FormerTailor: Customized Large Model Adaptation in Heterogeneous Cloud-Edge-Device Systems**"的代码

## 项目结构说明

```
├── data                      # 数据文件夹，用于存放数据集
├── draw                      # 绘图相关文件夹，用于存放生成的图像或绘图脚本
├── log                       # 日志文件夹，用于存放项目运行时生成的日志
├── model                     # 模型文件夹，用于存放训练好的模型或模型定义文件
├── moti                      # 动机相关的文件夹，用于存放动机分析的相关内容
│   └── origin_motiv          # 原始动机文件夹
│       └── log               # 存放原始动机分析过程中的日志
├── runs                      # 实验运行记录文件夹，用于存放实验运行的结果或中间文件
└── src                       # 源代码文件夹，包含项目的主要代码
    ├── backup                # 存放需要替换Transformer库的文件
    └── shell                 # Shell脚本文件夹，用于存放自动化任务的Shell脚本

```

## 预先条件

1. 下载并安装 `requirements.txt` 中要求的软件包：
   
    ```bash
    pip install -r requirements.txt
    ```
    
2. 下载预训练的 `vit-base-patch16-224` 模型，并将其放置在 `model` 目录下。

3. 手动下载 CIFAR-100 或 Stanford Cars 数据集，并将其放置在 `data` 目录下。

4. 使用 `src/backup` 中的文件替换 Transformers 库中相应的文件：

   - 定位到您的 Transformers 库安装目录。
   - 将 `src/backup` 中的文件复制到 Transformers 库的相应位置。

## 运行代码要求

要运行代码，请执行以下 shell 脚本：

1. 运行`cloud_pretrain.py`来获得预训练模型（注意调整`NAME`参数）。
    ```bash
    python cloud_pretrain.py 
    ```

2. 运行云端执行的第一个脚本获得**dynaViTw**
    ```bash
    ./run_cloud1.sh
    ```

3. 运行云端执行的第二个脚本获得**dynaViT**

   ```sh
   ./run_cloud2.sh
   ```

4. 运行边缘侧的NAS代码

   ```sh
   ./run_nas.sh
   ```

   

## 模型与训练集下载介绍

* [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

确保将下载的数据集放置在 `data` 目录中，模型放到`src`中。





