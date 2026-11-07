# 最终算例：F1 + 时间模型

## 三个点

| 名称 | 值 | 状态 |
|---|---|---|
| 0.10 m 平滑峰（主点） | (2.8750, 1.9250) m | 四种子等权；跨种子 0.0224 m |
| 原始格网 64k | (2.8975, 1.8808) m | 通过，但接近 0.20 m 阈值 |
| 辅助均值 | (2.9058, 1.9285) m | 辅助，不是最可能点 |

## 空间

| 区域 | 面积 | 阈值 |
|---|---:|---:|
| 模型内 90% | 1.1668 m² | 4.6976 |
| 校准 90% | 0.7659 m² | 3.0835 |

## 时间

- 中位到达时间：0.387688 s
- 原后验 90%：0.382739–0.394305 s
- 校准 90%：0.371435–0.403940 s（宽 32.51 ms）

- 入框概率：0.9667
- 未判定事件质量：0

## 限制

- spatial calibrated test coverage 88.6%; combined mechanism 82.8%
- time interval calibrated on synthetic development data; width 33.7 ms
- results are conditional on F1, fixed Cd=0.22, sigma=0.01 m, still air
- no real-world validation
