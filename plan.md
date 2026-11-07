# 方案

> 2026-11-08 更新。按 `1start-mathmodel` 的流程骨架，基于最终 F1 + 时间模型 + 平滑峰版本重写；不再沿用旧版“待实现”描述。

## 用户偏好

- 竞赛类型：UPC 2026 Problem B（University Physics Competition）
- 论文语言：英文（最终提交）；中间文档中文
- 子问题数量：题面没有编号子问题，按 **1 个连续预测任务** 处理
- 排版引擎：待定（进入论文阶段再选 Typst/LaTeX；当前不做论文）

## workflow

| 阶段 | skill | 状态 | 主要产物 |
|---|---|---|---|
| 1. 赛题分析与建模设计 | `2analysis-modeling` | 已完成最终版 | `reports/ANALYSIS_MODELING_REPORT.md` |
| 2. 编程实现与结果 | `3coding-visual` | 已完成核心版本 | `code/`、`results/` |
| 3. 流程图/架构图 | `4drawio` | 未开始，按论文需要决定 | `figures/*.drawio/.pdf` |
| 4. 竞赛论文撰写 | `5writing` | 未开始 | `paper/` |
| 5. 验证与验收 | `6verity` | 部分完成 | 测试、独立测试、最终核对 |

## 最终模型

- 主模型：**F1**
  - 方程：重力 + 二次阻力 + 有效横向力 \((0,b_y,b_z)\)；
  - 固定 \(C_D=0.22\)；
  - \(b_y\) 先验尺度 0.10，\(b_z\) 先验尺度 0.20；
  - 观测误差 \(\sigma=0.01\ \mathrm{m}\)，静风。
- 空间：
  - 模型内 90% 区域；
  - 校准 90% 区域（阈值 3.0835，独立 test 覆盖 88.64%）。
- 时间：
  - 全局 additive 误差项 \(q=15.1889\ \mathrm{ms}\)；
  - 独立 test 覆盖 96.10%，宽度 33.68 ms。
- 点：
  - 主点：0.10 m 固定带宽平滑峰 (2.875, 1.925) m；
  - 原始格网 64k：(2.8975, 1.8808) m，通过但 0.1833 m 接近阈值；
  - 辅助均值：(2.9058, 1.9285) m，**不是最可能点**。

## 代码入口

```bash
.venv/bin/python code/final_pipeline.py case
.venv/bin/python code/final_pipeline.py validate
.venv/bin/python code/final_pipeline.py check
```

核心模块：`config.py`、`physics.py`、`inverse.py`、`uncertainty.py`、`data.py`、`evaluate.py`、`final_config.py`、`final_pipeline.py`。
验证复现：`time_model_200.py`、`independent_200.py`、`f1_mode_stability.py`。

## 阶段产物

| 产物 | 位置 |
|---|---|
| 最终算例 | `results/final_case/final_case.json`、`final_case.md` |
| 时间模型 | `results/time_model_200/time_model_summary.json` |
| 空间校准/独立测试 | `results/independent_200/independent_summary.json` |
| 众数检查 | `results/f1_mode_stability/summary.json` |
| F0/F1 选择 | `results/round2_selection/summary.json` |
| 冻结配置 | `docs/superpowers/specs/2026-11-07-final-f1-freeze.md` |
| 代码说明 | `reports/FINAL_CODE_CHANGES.md` |

## 风险与关口

1. **空间覆盖未严格 90%**
   - 校准 test 88.64%，combined 82.81%；
   - 最终区域必须同时报告 1.1668 m²（保守）和 0.7659 m²（校准）。
2. **时间区间偏保守**
   - 33.7 ms、96.1% 覆盖；不能再靠缩小区间声称 90%。
3. **众数边界**
   - 原始格网 64k 通过但 0.1833 m 接近 0.20 m；
   - 平滑峰依赖 0.10 m 带宽，不是唯一连续众数。
4. **模型形式/misfit**
   - F1 的 MAP misfit 74/348；LS misfit 约 11.6%；
   - variable_drag/wind 仍是弱点。
5. **真实数据缺失**
   - 全部为 H 合成机制；外部有效性未验证。
6. **在线速度**
   - 完整后验约 7 s；未做快速版。

## 下一步优先级

1. 决定最终空间区域：0.7659 m²（校准，88.6%）或 1.1668 m²（保守，95.8%）；
2. 用新数据做一次最终独立确认（空间 + 时间）；
3. 若论文需要，补 S 对照和图表；
4. 再进入论文撰写与最终验收。
