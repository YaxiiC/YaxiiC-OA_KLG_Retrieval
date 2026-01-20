# OA KLGrade Subset Retrieval / 骨关节炎KL分级子集检索

## English

### Project Overview

This project predicts **Kellgren-Lawrence Grade (KLGrade 0-4)** from **3D knee MRI** using a **patient-specific radiomics subset retrieval** framework. Instead of gating each feature independently, we treat a **feature subset** as the unit and learn to **score and retrieve the best subsets** for each patient.

Current implementation:
- **Subset Scorer**: ranks candidate subsets for each patient
- **Token-Set KL Classifier**: predicts KLGrade from selected subsets
- **Few-step linear probe** (training-only): provides Reward B for scorer supervision
- **Budgeted retrieval** at inference (recall + rank + local search)

Legacy gate-based selection remains in `train_selector_klgrade.py`.

### Architecture (Current Implementation)

#### 1) Image Encoder (`models.py`)
- 3D CNN encoder
- Input: MRI `[B,1,D,H,W]`
- Output: `z_img ∈ R^d`

#### 2) Token-Set Encoder (`models.py`)
- Token = `(feature_id, value)` for each feature in a subset
- ID embedding + value MLP + DeepSets pooling
- Output: `z_set ∈ R^d`

#### 3) Subset Scorer (`models.py`)
- Input: `[z_img, z_set]`
- Output: `score ∈ R` (higher is better)

#### 4) Token-Set KL Classifier (`models.py`)
- Input: `[z_img, z_set]`
- Output: logits `[B,5]`

### Training Loss

```
Total Loss = L_cls + λ_rank × L_rank
```

- **L_cls**: KL cross-entropy on TopM ensemble logits
- **L_rank**: scorer regression loss vs Reward B (MSE)

### Training Strategy (Two Stages)

#### Stage 1: Warmup (epochs 1..T)
- Sample `N_subsets` randomly per patient
- Run probe on all `N_subsets` → Reward B
- Train scorer with `L_rank`
- Train classifier on TopM subsets by scorer (ensemble logits)

#### Stage 2: Main Training (after T)
- Sample `PoolSize` candidates per patient
- Scorer ranks pool
- Select `N_subsets` for probe (top-ranked + random exploration)
- Train scorer with probe rewards
- Train classifier on TopM subsets by scorer (ensemble logits)

### Data Pipeline

1. Segmentation: `nnunet_segmentation_inference.py`
2. Radiomics extraction: `torchradiomics_from_ROIs.py`
3. Radiomics format: **wide format** recommended (one row per case)
4. Training: `train_joint_scoring_kl.py`
5. Inference: `infer_budgeted_retrieval.py`

### File Structure

```
OA_KLG_Retrieval/
├── train_joint_scoring_kl.py     # Subset scorer + classifier training
├── infer_budgeted_retrieval.py   # Budgeted inference pipeline
├── train_selector_klgrade.py     # Legacy gate-based training
├── models.py                     # Model definitions
├── training_utils.py             # Training utilities (subset training, metrics)
├── data_loader.py                # Data loading and preprocessing
├── nnunet_segmentation_inference.py
├── torchradiomics_from_ROIs.py
├── environment.yml
├── output_train/
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
├── output_test/
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
└── training_logs/
```

### Usage

#### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate oa_klg_topk
```

#### 2. Train (Subset Retrieval)
```powershell
python train_joint_scoring_kl.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_train\radiomics_results_wide.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\subInfo_train.xlsx" `
    --outdir "training_logs_subset" `
    --k 15 `
    --n-subsets 32 `
    --top-m 4 `
    --pool-size 320 `
    --warmup-epochs 20 `
    --epochs 200 `
    --lambda-rank 0.1 `
    --exploration-ratio 0.2
```

#### 3. Inference (Budgeted Retrieval)
```powershell
python infer_budgeted_retrieval.py `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_test\radiomics_results_wide.csv" `
    --checkpoint "training_logs_subset\checkpoints\best.pth" `
    --scaler "training_logs_subset\checkpoints\scaler.joblib" `
    --outdir "inference_subset"
```

### Key Hyperparameters

- `--k`: subset size K
- `--n-subsets`: number of subsets for probe supervision
- `--top-m`: number of top subsets for classifier training
- `--pool-size`: candidate pool size after warmup
- `--warmup-epochs`: warmup epochs T
- `--lambda-rank`: weight for scorer ranking loss
- `--exploration-ratio`: fraction of random subsets after warmup
- `--probe-support`, `--probe-query`, `--probe-steps`, `--probe-lr`

### Output Files

#### Training Metrics (`logs/metrics.csv`)
- `epoch`, `train_loss`, `train_cls_loss`, `train_rank_loss`
- `val_loss`, `val_macro_f1`, `val_qwk`, `val_acc`

#### Predictions (`predictions_test.csv`)
- `case_id`, `pred_class`, `prob_0`..`prob_4`

#### Selected Subsets (`selected_subsets_test.json`)
- `case_id`, `final_top_indices` (K feature indices per subset)

### Evaluation Metrics

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- QWK (Quadratic Weighted Kappa)

---

## 中文

### 项目概述

本项目基于 **3D 膝 MRI** 预测 **KL 分级（0-4级）**，采用 **患者特异的影像组学子集检索** 框架。我们将“特征子集”作为选择单位，通过 **子集评分器** 排序，再用 **Token-Set 分类器** 预测 KL。

当前实现包含：
- **子集评分器**（Subset Scorer）
- **Token-Set KL 分类器**
- **few-step 线性 probe**（仅训练期，用于 Reward B）
- **固定预算推理**（召回 + 排序 + 局部搜索）

旧的 gate-based 方案仍保留在 `train_selector_klgrade.py`。

### 架构（当前实现）

#### 1) 图像编码器
- 3D CNN
- 输入：MRI `[B,1,D,H,W]`
- 输出：`z_img ∈ R^d`

#### 2) Token-Set 编码器
- token = `(feature_id, value)`
- ID embedding + value MLP + DeepSets pooling
- 输出：`z_set ∈ R^d`

#### 3) 子集评分器
- 输入 `[z_img, z_set]`
- 输出 `score ∈ R`

#### 4) KL 分类器
- 输入 `[z_img, z_set]`
- 输出 logits `[B,5]`

### 损失

```
Total Loss = L_cls + λ_rank × L_rank
```

- **L_cls**: TopM 子集集成后的 KL 交叉熵
- **L_rank**: scorer 与 Reward B 的回归损失

### 训练策略

#### 阶段1：Warmup
- 随机采样 `N_subsets`
- probe 产生 Reward B
- scorer 学习排序
- classifier 用 TopM 子集训练

#### 阶段2：主训练
- 采样 `PoolSize` 候选
- scorer 排序
- 选 `N_subsets` 进入 probe（高分 + 随机探索）
- classifier 用 TopM 子集训练（集成 logits）

### 数据流程

1. 分割：`nnunet_segmentation_inference.py`
2. Radiomics：`torchradiomics_from_ROIs.py`
3. 推荐 wide 格式（每行一个 case）
4. 训练：`train_joint_scoring_kl.py`
5. 推理：`infer_budgeted_retrieval.py`

### 文件结构

```
OA_KLG_Retrieval/
├── train_joint_scoring_kl.py
├── infer_budgeted_retrieval.py
├── train_selector_klgrade.py
├── models.py
├── training_utils.py
├── data_loader.py
└── ...
```

### 使用方法

#### 1. 环境
```bash
conda env create -f environment.yml
conda activate oa_klg_topk
```

#### 2. 训练（子集检索）
```powershell
python train_joint_scoring_kl.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_train\radiomics_results_wide.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\subInfo_train.xlsx" `
    --outdir "training_logs_subset"
```

#### 3. 推理（预算检索）
```powershell
python infer_budgeted_retrieval.py `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_test\radiomics_results_wide.csv" `
    --checkpoint "training_logs_subset\checkpoints\best.pth" `
    --scaler "training_logs_subset\checkpoints\scaler.joblib" `
    --outdir "inference_subset"
```

### 关键超参数

- `--k`, `--n-subsets`, `--top-m`, `--pool-size`
- `--warmup-epochs`, `--lambda-rank`, `--exploration-ratio`
- `--probe-support`, `--probe-query`, `--probe-steps`, `--probe-lr`

### 输出

- `logs/metrics.csv`
- `predictions_test.csv`
- `selected_subsets_test.json`

### 指标

- Accuracy, Balanced Accuracy, Macro F1, Weighted F1, QWK

