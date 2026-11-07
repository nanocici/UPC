# 最终代码修改说明

日期：2026-11-08（Asia/Shanghai）。
范围：只做最终版本整理与代码交付；**不改** `reports/ANALYSIS_MODELING_REPORT.md`、`plan.md`、`todo.md`；不写论文。

## 1. 最终新增文件

| 文件 | 作用 |
|---|---|
| `code/final_config.py` | 冻结常量：F1 先验、固定 Cd、时间 q、空间阈值、平滑峰/原始格网/辅助均值、种子与提议数 |
| `code/final_pipeline.py` | 唯一最终入口：`case` 生成指定算例，`validate` 生成验证表，`check` 校验冻结常量 |
| `tests/test_final_pipeline.py` | 检查冻结常量、时间/区域公式、三种点标签，防止把均值当众数 |
| `reports/FINAL_CODE_CHANGES.md` | 本说明 |

## 2. 最终运行链

```text
code/config.py
code/physics.py
code/inverse.py
code/uncertainty.py
code/data.py
code/evaluate.py
        +
code/final_config.py
code/final_pipeline.py
        +
tests/test_final_pipeline.py
```

`final_pipeline.py` 不改变 F 的方程：

- F1 = 同一 F 方程，`b_y` 先验 0.10；
- 空间校准 = 区域阈值后处理；
- 时间模型 = 区间后处理；
- 平滑峰 = KDE 估计层。

## 3. 现有文件修改情况

| 文件 | 本次是否修改 | 说明 |
|---|---|---|
| `code/physics.py` | 否 | 方程和事件未动 |
| `code/inverse.py` | 否 | 当前版本已含第一轮目标缩放/KKT认证；本轮未改 |
| `code/uncertainty.py` | 否 | 当前版本已输出后验、区域、时间区间；最终入口直接读 `summary` |
| `code/config.py` | 否 | 默认仍 F0；F1 由 `final_config.py` 显式传入，避免破坏旧结果 |
| `code/evaluate.py` | 否 | 评价指标未动 |
| `code/data.py` | 是（较早） | 允许 `split_fractions=(1,0,0)`，用于生成 development-only 数据；不是模型方程改动 |
| `code/round2_selection.py` | 是（较早） | 仅在预测记录中加入 `arrival_time_90_interval_s`，供验证/最终入口读取；不是核心模型改动 |

说明：`inverse.py` 和 `uncertainty.py` 当前版本包含第一轮修订的数值认证/输出有效性改动，这些是既有基线，不是本次新增的模型改动。

## 4. 冻结常量

| 常量 | 值 |
|---|---:|
| 模型 | F1（F，`b_y` 0.10，`b_z` 0.20） |
| 固定阻力 | `C_D=0.22` |
| 观测误差 | `sigma=0.01 m` |
| 时间 q | `0.015188910308210842 s`（15.1889 ms） |
| 空间校准阈值 | `3.083525450931334` |
| 平滑峰带宽 | `0.10 m` |
| 平滑峰 | `(2.875, 1.925) m` |
| 原始格网 64k | `(2.8975, 1.8808) m` |
| 辅助均值 | `(2.9057531544942057, 1.9285065612977923) m` |
| 算例种子 | `20261109` |
| 基础/总提议 | `2000 / 4000` |

## 5. 运行方法

从项目根目录：

```bash
.venv/bin/python code/final_pipeline.py case
.venv/bin/python code/final_pipeline.py validate
.venv/bin/python code/final_pipeline.py check
```

或一次运行：

```bash
.venv/bin/python code/final_pipeline.py all
```

输出：

```text
results/final_case/
  final_case.json
  final_case.md
  run_config.json
  summary.json
  validation.md
```

## 6. 最终输出字段

`final_case.json` 明确区分三类点：

- `primary_point`：0.10 m 固定带宽平滑峰；
- `raw_grid_64k`：新 64k 原始格网众数（通过但接近阈值）；
- `mean_auxiliary`：条件均值，仅辅助。

空间：

- `spatial.model_internal`：模型内 90% 区域（1.1668 m²）；
- `spatial.calibrated`：校准 90% 区域（0.7659 m²，独立 test 覆盖 88.6%）。

时间：

- `time.posterior_interval_s`：原后验 90%；
- `time.calibrated_interval_s`：校准 90%（约 32.5 ms）；
- `time.q_ms`：全局 additive q。

## 7. 验证表内容

`validation.md` 只读取已保存结果，不重新采样：

- 时间 q 与 time_model_200 test 覆盖；
- 空间校准阈值、原/校准区域覆盖与面积；
- combined 机制覆盖；
- 独立 test 时间覆盖；
- 众数 0.10 m 稳定性、原始格网 64k、历史失败；
- 最终算例主点/区域/时间/辅助均值。

## 8. 未修改/未完成

- **未修改** `reports/ANALYSIS_MODELING_REPORT.md`、`plan.md`、`todo.md`；
- 未写论文；
- 未加入自由 Cd；
- 未做真实数据验证；
- 空间校准 test 覆盖 88.6%、combined 82.8%，不能称完全 90% 校准；
- 时间区间宽 33.7 ms，偏保守；
- 在线快速版未做。

## 9. 测试与验收

- 完整项目测试：**127 项通过**（`unittest discover -s tests`）；
- `final_pipeline.py check` 返回 `time_q/spatial_threshold/smooth_mode/all = true`；
- `final_pipeline.py case` 已实际生成 `results/final_case/`；
- `final_pipeline.py validate` 已实际生成 `validation.md`；
- 未修改 `reports/ANALYSIS_MODELING_REPORT.md`、`plan.md`、`todo.md`。

最终新增测试：

```bash
.venv/bin/python -m unittest tests.test_final_pipeline -v
```

测试覆盖：

- 冻结常量；
- F1 先验；
- 时间区间公式；
- 空间区域面积公式；
- 三种点标签；
- 均值不得成为主点。


## 10. 清理记录

为保留最终可交付版本，删除以下不再进入最终运行/验证链的文件：

### 保留的验证依赖

以下文件不进入最终运行主链，但 `time_model_200`、`independent_200`、`f1_mode_stability` 仍依赖，保留：

- `code/development_regions.py`
- `code/model_diagnosis.py`
- `code/round2_selection.py`
- `code/mode_cause_check.py`

### 实际删除的代码

- `code/cd_sensitivity.py`
- `code/fdrag_profile.py`
- `code/time_cd_ablation.py`
- `code/diagnose_round2.py`
- `code/diagnosis_report.py`
- `code/first_revision.py`
- `code/first_revision_report.py`
- `code/mode_stability.py`
- `code/profile.py`
- `code/run.py`

### 删除的测试

- `test_first_revision.py`
- `test_mode_stability.py`
- `test_profile.py`
- `test_run.py`
- `test_time_cd_ablation.py`

### 删除的旧结果

- `results/controlled_diagnosis`
- `results/development_regions`
- `results/evaluation_demo`
- `results/first_revision_priors`
- `results/first_revision_validation`
- `results/inverse_development`
- `results/inverse_development_evaluation`
- `results/inverse_example`
- `results/model_diagnosis_existing`
- `results/mode_stability_example`
- `results/profile_example`
- `results/round2_diagnosis`
- `results/run_example_check`
- `results/uncertainty_example`
- `results/time_cd_ablation`
- `results/cd_sensitivity`
- `results/fdrag_profile`
- `results/mode_cause_check`
- `data/synthetic_demo`

### 保留的最终证据

- `results/final_case/`
- `results/time_model_200/`
- `results/independent_200/`
- `results/f1_mode_stability/`
- `results/round2_selection/`（F1 选择历史）
- `data/round2_development/`
- `data/time_model_200/`
- `data/independent_200/`
- `reports/` 下的报告
- `docs/superpowers/specs/` 下的预注册/冻结文件

本次未修改 `reports/ANALYSIS_MODELING_REPORT.md`、`plan.md`、`todo.md`。

## 11. 2026-11-08 稳定性结果清理

按用户要求，仅保留 `results/算例稳定性实验/重新运行_20261108/` 完整实验。删除上级目录旧的 `summary.json`、`kde_peaks.csv`、`grid_modes.csv` 及四个 `seed_*` 目录。`final_pipeline.MODE_PATH` 已指向重跑结果；旧报告改为当前报告入口。133项测试及冻结常量检查通过；模型和校准参数未变。

## 12. 2026-11-08 最终主点性能补充实验

完成已有test部分100源/400观测的批量稳定性及主点评分，结果位于`results/最终主点性能实验/`。保持当前F1及校准参数；使用经原积分器/步长减半验收的同方程C批量RK4加速。374组完成采样、6组拟合失败、20组观测不可用；保留1496个原始候选NPZ、共95,744,000条提议。预测全部封存后评分；139项测试通过。1cm层49条真实入框源均提供稳定平滑峰，F1/G中位误差0.2709/0.3739m；区域经验覆盖88.31%。原始格网在该层仅20/49通过，不能混用平滑峰与格网众数结论。详细限制见结果解读。另兼容稳定性结果移到中文根目录后的入口路径。
