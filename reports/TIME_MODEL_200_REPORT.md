# 路线 2：F-fixed 的 200 源时间误差模型

日期：2026-11-07（Asia/Shanghai）。
范围：新生成 200 个源；按机制分层，每类 20 个 calibration、20 个 test；只跑 F1 固定 \(C_D=0.22\)；共 800 个任务、3,088,000 次提议、2,392.9 s。原有 calibration/test 未读取，未加入自由 Cd。

## 1. 结论

1. **原始后验时间区间严重欠覆盖**：test 覆盖 53.2%，宽度中位仅 14.5 ms。
2. **全局 additive 时间误差模型达到 94.9% 覆盖**，宽度中位 33.7 ms；源级覆盖 94.9%，bootstrap 95% 区间 [90.7%, 98.1%]。
3. **全局 conformal 95.8%**，宽度 37.9 ms，偏保守。
4. **分机制 additive 只有 86.2%**，且 combined 仅 67.2%；每机制 20 个 calibration 源仍不足以稳定分层分位数。
5. **建议采用 F-fixed + 全局 additive 时间误差模型作为候选**；它修正了时间欠覆盖，但宽度约为原区间的 2.3 倍，且尚未做最终独立测试。

## 2. 数据与运行

| 项目 | 值 |
|---|---:|
| 源 | 200（calibration 100 / test 100） |
| 机制 | 每类 40，calibration/test 各 20 |
| 观测 | 800（4 档噪声） |
| 真实直接过线 | calibration 312 / test 312 |
| 任务 | 800（F1 fixed Cd） |
| 提议 | 3,088,000 |
| 墙钟 | 2,392.9 s（约 39.9 min） |

## 3. test 覆盖与宽度

| 区间方案 | 覆盖 | 宽度中位 | 说明 |
|---|---:|---:|---|
| 原始后验 | 53.2% | 14.5 ms | 基线 |
| 全局 conformal | 95.8% | 37.9 ms | ±q_global |
| 分机制 conformal | 95.2% | 34.9 ms | ±q_mech |
| **全局 additive** | **94.9%** | **33.7 ms** | \(\sqrt{h^2+q_{\rm global}^2}\) |
| 分机制 additive | 86.2% | 31.4 ms | combined 仅 67.2% |

参数：

- \(q_{\rm global,conformal}=19.0\) ms；
- \(q_{\rm global,additive}=15.2\) ms。

源级（global additive）：

- 78 个 test 源；
- 源级覆盖 94.9%；
- 源级 bootstrap 95% 区间 [90.7%, 98.1%]。

## 4. 分机制

全局 additive：

| 机制 | n | 覆盖 | 宽度中位 |
|---|---:|---:|---:|
| constant | 56 | 91.1% | 34.5 ms |
| variable_drag | 56 | 100% | 33.6 ms |
| spin | 60 | 88.3% | 34.4 ms |
| wind | 76 | 94.7% | 32.5 ms |
| combined | 64 | 100% | 33.4 ms |

- 所有机制 ≥88.3%，没有再出现 40%–60% 的塌陷；
- variable_drag/combined 偏保守，spin 略低，但在 85%–95% 附近；
- 分机制 additive 的 combined 67.2% 说明 **不要用当前 20 源/机制的分位数做分层模型**；全局误差项更稳。

## 5. 解释

- 原始区间只传播了参数不确定性，缺少约 15 ms 的模型误差项；
- 加上全局误差项后，宽度从 14.5 ms 增到 33.7 ms，覆盖从 53.2% 升到 94.9%；
- 这是“把误差预算补足”，不是“时间预测变准”：时间点误差没有变化；
- 94.9% 略高于名义 90%，偏保守；宽度 33.7 ms 是代价；
- 该模型仍是 **development calibration/test**，不是最终独立测试；test 源来自同一 H 生成机制。

## 6. 决策

**采用 F-fixed + 全局 additive 时间误差模型作为下一阶段候选。**

- 不再加自由 Cd；
- 不再使用分机制 additive 作为主模型；
- 空间仍使用第二轮 F1 结果；
- 时间区间在最终论文中必须标注“校准后的开发集区间”，不能写成外部 90% 保证。

## 7. 下一步

1. 把该时间模型接入最终 F1 输出；报告时间覆盖、宽度和源级区间；
2. 在独立测试集上重复一次，确认 94.9% 不是这批 calibration/test 的偶然；
3. 检查时间宽度 33.7 ms 是否影响门将应用；若要求更窄，需要更多信息或更强模型，不能靠缩小区间；
4. 仍保持 misfit 的 `map/ls` 分开报告；
5. 最后写论文时，把“时间点误差”和“时间区间覆盖”分开陈述。

## 8. 复现

```bash
.venv/bin/python code/time_model_200.py prepare
.venv/bin/python code/time_model_200.py run --workers 6
.venv/bin/python code/time_model_200.py score
```

输出：

- `data/time_model_200/`：200 源、calibration/test 各 100；
- `results/time_model_200/predictions.jsonl`：封存 F1 预测；
- `results/time_model_200/time_model_summary.json`：q、覆盖、分机制、源级区间；
- 本报告：`reports/TIME_MODEL_200_REPORT.md`。
