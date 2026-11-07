# 最终主点性能实验

流程分为两个阶段，复用同一批后验候选，不重复采样：

1. 实验二批量检查：每组公开观测四个种子、各64k提议，记录峰位置、采样诊断和稳定性状态。没有稳定主点的案例仍保留。
2. 实验三评分：400组观测的预测全部封存后，才由评分程序读取测试真值，比较F1主点和G，并统计空间区域的经验覆盖与面积。

输入是已有independent_200的test部分：100条独立源射门、四档噪声、共400组观测。模型及校准参数冻结；这是已有数据上的补充评价，不是新的盲测。

| 文件或目录 | 内容 |
|---|---|
| `cases/<case_id>/candidates_<seed>.npz` | 无损压缩原始候选：参数、目标和提议对数密度、权重、事件、位置和时间；零权重候选保留 |
| `cases/<case_id>/summary_<seed>.json` | 单种子32k/64k采样诊断 |
| `cases/<case_id>/prediction.json` | 不含真值的峰、稳定性状态、失败原因、G预测与区域 |
| `predictions.jsonl`、`prediction_manifest.json` | 评分前封存的400组预测及指纹 |
| `per_case.csv` | 逐案例真值评分与失败记录 |
| `summary.json` | 总体、1cm主要噪声层、各噪声/机制及配对bootstrap |
| `FINAL_POINT_EVALUATION_REPORT.md` | 实验方法、性能表、结论和限制 |
| `point_errors.png` | 共同成功子集的误差CDF与配对散点图 |
| `coverage_and_stability.png` | 分层覆盖率和稳定输出比例 |
| `numerical_validation.json` | 不读取真值的原积分器对照及步长减半验收 |
| `run_config.json`、`source/` | 设置、输入指纹与源码快照 |

主带宽0.10m；0.05/0.20m和原始格网另作诊断。入框条件峰误差只在真实入框射门上评价；区域覆盖在真实直接过门线射门上评价。两者样本条件不同，不能混用。只比较共同成功子集的误差时，同时报告稳定提供率、失败数及G在全部可用案例上的表现。

运行入口：`code/final_point_evaluation.py`，支持`prepare`、`run`、`score`和`all`。`run`复用已完成案例和候选，`score`不重新采样。C批量积分仅加速当前F方程，并通过原积分器及步长减半的数值验收。

已完成。主要入口：[结果解读](INTERPRETATION.md)、[完整报告](FINAL_POINT_EVALUATION_REPORT.md)。`stability.csv`和`stability_summary.json`只使用封存预测生成，不需要真值。

注意：NPZ中的`goal_yz_m`仅在event=1/2时表示门线坐标；event=3时保留的是首个接地事件坐标，供数值核验，不作为门线预测。event编号与拒绝/未决定义见source/batch_f_sampler.py。
