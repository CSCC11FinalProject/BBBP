# KNN Classifier — Changelog

## [1.0.0] - 2026-03-11

### Added

- **`preprocess.py`** — 数据预处理与特征工程模块
  - 加载已预处理的 `dataset/BBBP.csv`（化学描述符已由 `dataset/process.py` 生成）
  - 调用 `dataset/utils.py` 的 `get_morgan_fingerprint()` 将 SMILES 转为 2048-bit Morgan Fingerprint
  - 拼接 Morgan FP (2048) + 化学描述符 (7) → 2055 维特征矩阵
  - 化学描述符 Pearson 相关矩阵分析，识别 |r| > 0.85 的冗余特征对
  - 相关矩阵热力图保存至 `plots/correlation_matrix.png`
  - `StandardScaler` 标准化（仅在训练集上 fit，防止数据泄露）
  - Stratified 80/10/10 train/val/test 划分（`SEED=67`，与 MPNN 一致）
  - 对外暴露 `load_and_preprocess()` 函数，供其他模块调用

- **`tuning.py`** — Optuna 超参数调优
  - 搜索空间：`n_neighbors` (1–50 奇数), `weights` (uniform/distance), `metric` (euclidean/manhattan/minkowski), `p` (1–5)
  - 优化目标：验证集 AUC-ROC（direction="maximize"）
  - 50 次 trial，结果保存至 `best_params.json`

- **`train.py`** — 最终模型训练与保存
  - 从 `best_params.json` 加载调优后的超参数（若无则用默认值）
  - 在训练集上 fit `KNeighborsClassifier`
  - 保存模型和 scaler 至 `checkpoints/`（joblib 格式）
  - 输出测试集 AUC-ROC 和 F1 快速评估

- **`evaluate.py`** — 全面测试集评估与可视化
  - 核心指标：AUC-ROC, F1, Precision, Recall, Specificity
  - 混淆矩阵热力图 → `plots/confusion_matrix.png`
  - ROC 曲线 → `plots/roc_curve.png`
  - False Positive 误分类分析：
    - 对比 FP vs TN 在 LogP/TPSA/MW 上的均值差异
    - 保存详情至 `plots/false_positives.csv` 和 `false_positive_summary.csv`
    - 使用 RDKit 绘制 FP 分子结构图 → `plots/false_positives_structures.png`

- **`checklist.md`** — 详细实施计划与任务清单

### Design Decisions

- **与 MPNN 保持一致**：SEED=67, 80/10/10 划分, AUC-ROC 为主优化指标, 评估输出格式对齐
- **特征表示**：Morgan FP 提供分子拓扑信息，化学描述符提供物理化学性质，两者互补
- **相关矩阵仅分析 7 个描述符**：Morgan FP 是 2048 维稀疏二值向量，不适合 Pearson 相关分析
- **Stratified 划分**：数据集存在类别不平衡，stratify 确保各子集类别比例一致
- **Scaler fit 仅用训练集**：严格防止数据泄露，保证评估的公正性
