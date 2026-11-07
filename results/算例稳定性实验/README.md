# 算例稳定性实验文件说明

本目录只保留 `重新运行_20261108/` 这一套完整实验结果，包含按当前 F1 重新生成的原始候选、配置与源码快照、统计表、密度数值、图表和报告。

原根目录的三个统计文件和四个 `seed_*` 目录已于2026-11-08按用户要求删除。最终入口与测试使用本次重新运行的汇总。报告中保留的旧数值仅用于历史对照，不依赖被删除的文件。

新结果的入口是 [STABILITY_REPORT.md](重新运行_20261108/STABILITY_REPORT.md)。

| 新结果文件 | 用途 |
|---|---|
| `seed_*/candidates.jsonl.gz` | 四个种子分别保存的全部64k提议，包含拒绝/零权重候选 |
| `seed_*/summary.json` | 各种子的采样和事件诊断 |
| `summary.json` | 合并峰、带宽/种子/采样量稳定性及原始格网对照 |
| `kde_peaks.csv` | 单种子与合并的各档平滑峰 |
| `grid_modes.csv` | 各格网宽度、平移与采样量下的原始众数 |
| `sampling_convergence.png` | 采样量变化对应的峰位置 |
| `cross_seed_peaks.png` | 主带宽下的跨种子峰位置 |
| `bandwidth_sensitivity.png` | 三个固定带宽的峰位置对照 |
| `kde_density.png` | 直接过线与入框条件下的合并平滑密度 |
| `kde_density.npz` | 密度图对应的数值矩阵 |
| `run_config.json`、`source/` | 输入、设置、版本及源码快照 |
| `execution.json`、`artifact_manifest.json` | 执行记录及文件校验值 |

模型、先验、阻力、观测误差、主带宽和判据均保持冻结。新实验用于完整复现和增加可检查证据，不通过选择种子或改变带宽获得更好结果。稳定性不等于真实预测精度。
