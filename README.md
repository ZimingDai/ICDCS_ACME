
# ACME: Adaptive Customization of Large Models via Distributed Systems

[![切换为英文版](https://img.shields.io/badge/GitHub-Chinese-blue?logo=github)](./README_CN.md)

This repository contains the official implementation of the ICDCS 2025 accepted paper **"ACME: Adaptive Customization of Large Models via Distributed Systems"**.

## 📚Overview

ACME (Adaptive Customization of Large Models via Distributed Systems) is a framework designed to adaptively customize large Transformer-based models across distributed systems. It addresses major deployment challenges such as **model-device mismatch, resource constraints, and data heterogeneity**. ACME proposes a **bidirectional single-loop distributed system** that enables progressive model customization through collaboration between the cloud, edge servers, and devices. By separating **backbone generation** and **data-aware header refinement**, and avoiding direct transmission of local data, ACME achieves **high accuracy with minimal communication cost**, enabling efficient and personalized model deployment at scale.

## 📁 Project Structure

```
├── data/                    # Datasets (place CIFAR-100 or Stanford Cars here)
├── draw/                    # Figure generation scripts and outputs
│   ├── draw_challenge_motiv/  # Challenge-related visualizations
│   └── draw_experi/           # Experiment result visualizations
├── log/                     # Logs during training/experiments
├── model/                   # Model definitions and pretrained models
├── moti/                    # Motivation analysis materials
│   └── origin_motiv/
│       └── log/             # Original motivation experiment logs
├── runs/                    # Experimental results and intermediate files
└── src/                     # Main source code
    ├── backup/              # Customized files to override parts of Transformers
    ├── shell/               # Shell scripts for automated execution
    └── third_part/          # ⚠️ Independent implementation for Part III of the paper 
```

## 🛠️ Prerequisites

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the pretrained model:

   - [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
   - Place it in the `model/` directory.

3. Download datasets:

   - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
   - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
   - Place them in the `data/` directory.
  
    > ⚠️ **Note on Stanford Cars Dataset**  
    > The official source of the Stanford Cars dataset is currently **unavailable or corrupted**, causing failures when using:
    >
    > ```python
    > torchvision.datasets.StanfordCars(...)
    > ```
    >
    > **Solution:**  
    > 1. Download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset).  
    > 2. Extract and organize it into the following folder structure:
     

4. ⚠️ **Important: Overwriting Transformers Source Code**

   To enable customized behavior, you must **manually replace** parts of the Transformers library:

   - Locate the installation path of your local Transformers library (typically `site-packages/transformers/`)
   - Copy and **overwrite** the corresponding files from `src/backup/` into the above directory

   > ⚠ Warning: This operation modifies the default implementation of Transformers. Proceed only if you understand the changes, and it is highly recommended to use a virtual environment to avoid affecting other projects.
   
5. ⚠️ **Note: `src/third_part/` is a standalone project for Part III of the paper**


## 🚀 How to Run

1. Run cloud pretraining:

   ```bash
   python cloud_pretrain.py
   ```

2. Run cloud script for **DynaViTw**:

   ```bash
   bash ./src/shell/run_cloud1.sh
   ```

3. Run cloud script for **DynaViT**:

   ```bash
   bash ./src/shell/run_cloud2.sh
   ```

4. Run edge-side NAS:

   ```bash
   bash ./src/shell/run_nas.sh
   ```

5. ⚠️ To run the fine-grained personalization in Part III, go to `src/third_part/` and execute:

   ```bash
   cd src/third_part
   bash run.sh
   ```

## 📦 Model and Dataset Links

| Type | Name | Link |
|------|------|------|
| Model | vit-base-patch16-224 | [Download from Hugging Face](https://huggingface.co/google/vit-base-patch16-224) |
| Dataset | CIFAR-100 | [Official CIFAR Site](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Dataset | ~~Stanford Cars~~ | [~~Stanford AI Lab~~](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) unavailable|

