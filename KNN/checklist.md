# KNN Classifier — Implementation Checklist

参照 MPNN 的项目结构，KNN 部分将创建以下文件，每个文件的职责和细分任务如下。

---

## 1. `preprocess.py` — 数据预处理与特征工程

负责将原始 CSV 数据转换为 KNN 可用的数值特征矩阵，并完成数据划分。

- [ ] **1.1 加载数据集**
  - 读取 `dataset/BBBP.csv`，丢弃含 NaN 的行（与 MPNN 一致）
  - 提取标签列 `p_np` 作为 `y`

- [ ] **1.2 生成 Morgan Fingerprint**
  - 调用 `dataset/utils.py` 中的 `get_morgan_fingerprint(smiles)` 将每个 SMILES 转换为 2048 维 bit vector
  - 处理转换失败的行（`None`），将其丢弃

- [ ] **1.3 拼接化学描述符特征**
  - 从 CSV 中提取已有的 7 个描述符：`LogP, TPSA, MW, HBA, HBD, RotatableBonds, Charge`
  - 与 Morgan Fingerprint 拼接，形成完整特征矩阵（2048 + 7 = 2055 维）

- [ ] **1.4 相关矩阵分析与特征选择**
  - 计算 7 个化学描述符之间的相关矩阵（Pearson correlation）
  - 绘制相关矩阵热力图，保存至 `KNN/plots/correlation_matrix.png`
  - 识别高度相关的特征对（|r| > 0.85），决定是否移除冗余特征并记录理由
  - 注意：Morgan Fingerprint 的 2048 维不参与相关矩阵分析（维度太高且为稀疏二值向量）

- [ ] **1.5 特征缩放**
  - 使用 `StandardScaler` 对所有特征进行标准化（均值=0，标准差=1）
  - **仅在训练集上 fit**，然后 transform 训练集、验证集和测试集（防止数据泄露）

- [ ] **1.6 数据集划分**
  - 按 80% / 10% / 10% 划分 train / val / test（与 MPNN 一致）
  - 使用 `SEED = 67` 保证可复现性（与 MPNN 一致）
  - 使用 `stratify` 选项确保类别分布一致

- [ ] **1.7 封装为可复用函数**
  - 提供 `load_and_preprocess()` 函数，返回 `X_train, X_val, X_test, y_train, y_val, y_test` 以及 scaler 对象
  - 其他文件（tuning.py、train.py、evaluate.py）均调用此函数获取数据

---

## 2. `tuning.py` — 超参数调优

使用 Optuna（与 MPNN 一致）在验证集上搜索最优超参数。

- [ ] **2.1 定义搜索空间**
  - `n_neighbors`（k 值）：搜索范围 [1, 50] 之间的奇数
  - `weights`：`uniform` vs `distance`（距离加权）
  - `metric`：`euclidean`, `manhattan`, `minkowski`
  - `p`（当 metric=minkowski 时）：[1, 5] 范围内的整数

- [ ] **2.2 实现 Optuna objective 函数**
  - 调用 `preprocess.py` 获取数据
  - 用 trial 采样的超参数创建 `KNeighborsClassifier`
  - 在训练集上 fit，在验证集上评估
  - 以 val AUC-ROC 为优化目标（与 MPNN 一致，direction="maximize"）

- [ ] **2.3 运行调优并输出结果**
  - 运行足够数量的 trials（如 50 次，KNN 比 MPNN 快得多）
  - 打印最佳超参数和对应的 val AUC-ROC
  - 将最佳参数保存到 `KNN/best_params.json` 以供后续使用

---

## 3. `train.py` — 使用最优超参数训练最终模型

- [ ] **3.1 加载最佳超参数**
  - 从 `KNN/best_params.json` 读取调优结果
  - 如果文件不存在，使用合理的默认超参数

- [ ] **3.2 训练最终 KNN 模型**
  - 使用最优超参数创建 `KNeighborsClassifier`
  - 在训练集上 fit

- [ ] **3.3 保存模型**
  - 使用 `joblib` 将训练好的模型保存到 `KNN/checkpoints/knn_model.joblib`
  - 同时保存 scaler 对象到 `KNN/checkpoints/scaler.joblib`

- [ ] **3.4 输出基本测试指标**
  - 在测试集上快速评估：打印 Test AUC-ROC 和 Test F1

---

## 4. `evaluate.py` — 全面评估与可视化

参照 MPNN 的 `evaluate.py`，对 KNN 模型进行详细的测试集评估。

- [ ] **4.1 加载模型和数据**
  - 从 checkpoint 加载 KNN 模型和 scaler
  - 调用 `preprocess.py` 获取测试数据

- [ ] **4.2 计算核心指标**
  - AUC-ROC
  - F1 Score
  - Precision / Recall / Specificity
  - Confusion Matrix 各项 (TP, FP, TN, FN)

- [ ] **4.3 绘制混淆矩阵**
  - 使用 seaborn heatmap，标签为 "BBB-" / "BBB+"
  - 保存至 `KNN/plots/confusion_matrix.png`

- [ ] **4.4 绘制 ROC 曲线**
  - 绘制 FPR vs TPR 曲线，标注 AUC 值
  - 保存至 `KNN/plots/roc_curve.png`

- [ ] **4.5 误分类分析（False Positives）**
  - 找出 false positives（实际 BBB- 但预测为 BBB+）
  - 比较 false positives 与 true negatives 在化学描述符上的差异
  - 保存分析结果到 CSV
  - 可选：使用 RDKit 绘制 false positive 分子结构图

- [ ] **4.6 打印完整评估报告**
  - 格式化输出所有指标（与 MPNN evaluate.py 的输出风格对齐）

---

## 项目约定（与 MPNN 保持一致）

| 约定项 | 值 |
|---|---|
| 随机种子 | `SEED = 67` |
| 数据划分 | 80% train / 10% val / 10% test |
| 主要优化指标 | AUC-ROC |
| 评估指标 | AUC-ROC + F1 Score |
| 数据集路径 | `../dataset/BBBP.csv`（相对于 KNN 文件夹） |
| 图表保存路径 | `KNN/plots/` |
| 模型保存路径 | `KNN/checkpoints/` |

---

## 文件依赖关系

```
dataset/utils.py  ──→  preprocess.py  ──→  tuning.py  ──→  train.py  ──→  evaluate.py
dataset/BBBP.csv  ──↗                       ↓
                                        best_params.json
```

执行顺序：`preprocess.py`（可单独运行验证）→ `tuning.py` → `train.py` → `evaluate.py`
