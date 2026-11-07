# UPC 2026 Problem B：罚点球轨迹预测

根据罚球后 0.05 s 和 0.10 s 的三维位置，预测球心穿过门线的位置及不确定性。

## GitHub 同步范围

代码、数据、报告、图表及汇总结果纳入版本管理。`results/` 下约 10.9 GB 的原始候选 NPZ 文件保留本地，由 `.gitignore` 排除；依赖这些候选文件的分析需先按对应实验配置重新生成，或使用本地原始文件。

## 当前进度

最初规划的七个 Python 文件和阻力剖面、算例众数检查已实现，另完成 `code/development_regions.py` 的开发区域检验。项目 130 项检查通过；36 个开发源射门、144 条观测的 S/F 后验已全部处理。名义 90% 区域实际覆盖 S 为 44.64%、F 为 94.64%，F 区域面积中位数约为 S 的 6.89 倍；共同有效案例未显示 F 点误差有明确优势，见 [开发区域检查报告](reports/DEVELOPMENT_REGIONS_CHECK_REPORT.md)。这些是模拟开发结果，模型/先验冻结、校准和独立测试仍待完成。

算例众数检查已完成，但名义 S/F 都未通过完整众数稳定性标准，见 [众数检查报告](reports/MODE_STABILITY_CHECK_REPORT.md)；区域覆盖结果不替代该限制。

| 文件 | 内容 |
|---|---|
| [plan.md](plan.md) | 实施计划和模型分工 |
| [todo.md](todo.md) | 当前进度与后续任务 |
| [建模报告](reports/ANALYSIS_MODELING_REPORT.md) | 物理模型、推断和验证设计 |
| [code/config.py](code/config.py) | 题面数据、工作参数和共同类型 |
| [code/run.py](code/run.py) | 最小统一入口，默认仅运行题目 F 算例 |
| [code/profile.py](code/profile.py) | S/F 的 LS/MAP 阻力重拟合剖面及固定参数前向对照 |
| [code/mode_stability.py](code/mode_stability.py) | 题目算例的采样量、种子、格网与先验/误差敏感性检查 |
| [code/development_regions.py](code/development_regions.py) | 开发组计时、批量预测、评分与区域/面积/失败率配对比较 |
| [代码说明](code/README.md) | 三人协作接口与运行方法 |

## 检查共享配置

Python 3.9 或更高版本，从项目根目录执行：

Mac：

```bash
python3 code/config.py
```

Windows：

```powershell
py -3 code/config.py
```

配置文件仅使用 Python 标准库。输出 `Config check passed` 表示配置检查通过。数值模块使用 NumPy、SciPy 和 Matplotlib，依赖见 requirements.txt。

## 一条命令跑题目算例

已创建 `.venv` 并安装 requirements.txt 中的依赖后，从项目根目录运行。

Mac：

```bash
.venv/bin/python code/run.py --plot
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\run.py --plot
```

默认仅算 F：调用反演，再调用不确定性程序。自动在 results 下新建带时间的目录，run_config.json 保存设置，logs 保存日志，run_summary.json 汇总执行状态和模块诊断。`completed` 表示命令流程执行完成；若有众数不稳定等问题，另标 `review_required`。详细选项见 code/README.md。


诊断第1、2步已完成：方向诊断、同模型恢复、30组配对机制消融及匹配先验检查。完整结果见 [诊断报告](reports/MODEL_DIAGNOSIS_REPORT.md)，新增入口为code/model_diagnosis.py；137项测试通过。


第一轮修订已完成：优化认证、分项输出、固定上限追加预算及F先验对照；145项测试通过。修复F0作为基线，by=0.10保留候选，0.05不采用；见[第一轮修订报告](reports/FIRST_REVISION_REPORT.md)。


第二轮 F0/F1 选择已完成：新增100个development源、400条观测、800个任务；F1在预注册五项主条件上全部通过，面积中位数1.352 m²、覆盖94.54%、点误差中位数0.295 m，决策为采纳F1作为空间候选。名义90%时间区间仅覆盖约50.6%，尚不可发布；S对照暂缓，校准/测试未开启。完整结果见[第二轮选择报告](reports/ROUND2_SELECTION_REPORT.md)；当前155项测试通过。

第二轮诊断已完成：时间区间的问题是“误差下限”未计入（σ=0.005 时宽度5.7 ms < MAE约5 ms，覆盖26%），机制间有-7～+4 ms系统偏差；misfit 主要是先验收紧造成的 MAP 残差（σ=0.05 时 MAP 32.2% vs LS 11.6%），真正 LS misfit 约11.6%，集中在变阻力/风。阻力是成分之一，但不足以单独解释时间欠覆盖。见[诊断报告](reports/ROUND2_DIAGNOSIS_REPORT.md)。

固定Cd消融与时间校准已完成：在共同334例上，把Cd固定到每个源的有效真实值使时间MAE从5.34 ms降到2.43 ms、P90从15.09 ms降到8.73 ms；固定剖面最优0.37375反而使MAE升到14.09 ms。时间误差项把test覆盖从51.1%提高到83.3%，但仍未到90%，分机制条件化无额外收益。结论：阻力是时间误差主要成分之一，但不能全局固定Cd；时间模型需要更大校准集。见[固定Cd与时间报告](reports/TIME_CD_ABLATION_REPORT.md)；158项测试通过。

F-drag Phase 1 完成：F-est/F-int 未通过筛选。两帧 profile 估计中 238/334 选中 Cd=0.22，F-est 时间 MAE 5.43 ms（F-fixed 5.29），F-int 5.27 ms；空间点误差略差，F-int profile 空间椭圆退化。结论：阻力 oracle 收益存在，但两帧无法可靠识别 Cd；不采纳 F-drag，保留 F-fixed。见[F-drag Phase 1 报告](reports/FDRAG_PHASE1_REPORT.md)。

路线2时间模型完成：200源（calibration/test各100，每机制20）F1固定Cd，800任务/3.088M提议。原始时间区间覆盖53.2%、宽14.5ms；全局additive误差项覆盖94.9%、宽33.7ms，源级95%区间[90.7,98.1]；分机制additive仅86.2%。采用F-fixed+全局additive时间模型作为候选，仍属开发校准/测试。见[时间模型报告](reports/TIME_MODEL_200_REPORT.md)。

指定算例冻结输出（F1 + 校准时间模型）：主点为0.10 m固定带宽平滑峰 (2.875,1.925) m（四种子等权，跨种子0.022 m）；原始格网64k众数(2.8975,1.8808) m通过但0.1833 m接近阈值，历史4k/8k/16k失败保留；辅助均值(2.9058,1.9285) m。90%模型内区域1.1668 m²，入框概率96.67%，到达中位0.387688 s，校准时间区间[0.371435,0.403940] s（宽32.51 ms）。见[最终算例报告](reports/FINAL_CASE_F1_REPORT.md)、[众数报告](reports/F1_MODE_STABILITY_REPORT.md)与[冻结补充](docs/superpowers/specs/2026-11-07-final-f1-mode-addendum.md)。

独立测试完成：新200源（100 calibration/100 test，每机制20/20），F1固定Cd，800任务/3.028M提议。空间校准阈值q=3.0835：calibration覆盖89.9%、test覆盖88.6%（源级95%区间84.7–92.2），面积中位1.475→0.983 m²；combined机制仅82.8%。时间冻结q=15.1889ms在test覆盖96.1%（源级91.6–99.7）、宽33.7ms；点误差中位0.318m/P90 0.615m。见[独立测试报告](reports/INDEPENDENT_TEST_REPORT.md)。
