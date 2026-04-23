# Digit Preprocessing Project (`digits4000.mat`)

这个项目用于复现并支撑你的课程 report，目标是系统比较：

- 预处理：`raw / standard / minmax / pca / standard_pca`
- 分类器：`logistic / linear_svm / knn`
- 评估内容：clean accuracy、Gaussian noise robustness、confusion matrix、failure cases、PCA 维度曲线

目前代码已改成更稳妥的 **scikit-learn 分类器实现**：

- `logistic` → `sklearn.linear_model.LogisticRegression`
- `linear_svm` → `sklearn.svm.LinearSVC`
- `knn` → `sklearn.neighbors.KNeighborsClassifier`

这样更符合课程报告里的“logistic / linear SVM / Euclidean kNN”设定，也比原先的手写 PyTorch 版本更稳定、更快、更容易复现。

## 1. 环境安装

建议 Python 3.10+。

```bash
cd digit_preprocess_fixed
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 旧版 README 里的 `cd digit_preprocess_pytorch` 不对；请使用当前项目目录名。

## 2. 数据放置

课程数据默认放在：

```text
data/digits4000.mat
```

也可以通过 `--mat_path` 显式指定。

## 3. 推荐执行顺序

### A. 检查数据是否读对

```bash
python main.py inspect \
  --config configs/base.yaml \
  --mat_path data/digits4000.mat \
  --output_dir outputs/course_digits_full
```

### B. 跑完整实验

```bash
python main.py run \
  --config configs/base.yaml \
  --mat_path data/digits4000.mat \
  --output_dir outputs/course_digits_full
```
[saas-nexus-1776844650393.md](../../Download/saas-nexus-1776844650393.md)
### C. 生成 report 所需图表和表格

```bash
python main.py report \
  --config configs/base.yaml \
  --mat_path data/digits4000.mat \
  --output_dir outputs/course_digits_full
```

或者直接：

```bash
bash scripts/run_all.sh data/digits4000.mat outputs/course_digits_full configs/base.yaml
```

## 4. 与 report.tex 对应的核心输出

### 图像

- `figures/dataset_preview.png`
- `figures/dimension_accuracy_curve.png`
- `figures/robustness_curve.png`
- `figures/confusion_best_clean.png`
- `figures/confusion_best_robust.png`
- `figures/failure_cases_best_clean.png`
- `figures/failure_cases_best_robust.png`

同时也会额外保留：

- `figures/accuracy_vs_pca.png`
- `figures/robustness_curves_best_per_model.png`

### 表格

- `tables/clean_summary_by_config.csv`
- `tables/main_accuracy_table.csv`
- `tables/main_accuracy_table.tex`
- `tables/main_accuracy_table_long.csv`
- `tables/noise_summary_by_config.csv`
- `tables/robustness_drop_table.csv`
- `tables/robustness_highest_noise_summary.csv`
- `tables/robustness_highest_noise_summary.tex`
- `best_configs.json`

## 5. smoke test

```bash
python tools/make_synthetic_digits_mat.py --out data/synthetic_digits4000.mat
python main.py inspect --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
python main.py run --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
python main.py report --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
```

或：

```bash
bash scripts/smoke_test.sh
```

## 6. technical correctness

- 预处理只在训练集 `fit`
- 测试集只做 `transform`
- 超参数只通过训练集内部 CV 选择
- test set 只用于最终评估
- Gaussian noise 只加在测试集上做 robustness 分析
- 每个 official trial 独立评估，再汇总 `mean ± std`

## 7. 已知说明

1. `raw` 表示只做基础像素范围归一化到 `[0,1]`。
2. `pca` 表示直接 PCA；`standard_pca` 表示先 standardize 再 PCA。
3. `report.tex` 里如果还保留 `filecite...` 这类标记，需要手动删掉，否则 LaTeX 会报错。
4. 代码会生成 csv/tex 结果文件，但不会自动改写你的 `report.tex`。
