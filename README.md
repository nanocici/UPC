# UPC 2026 Problem B：罚点球轨迹预测

根据罚球后 0.05 s 和 0.10 s 的三维位置，预测球心穿过门线的位置及不确定性。

## 当前进度

建模方案和共享配置已建立。运动求解、参数反演和模型验证尚待实现。

| 文件 | 内容 |
|---|---|
| [plan.md](plan.md) | 实施计划和模型分工 |
| [todo.md](todo.md) | 当前进度与后续任务 |
| [建模报告](reports/ANALYSIS_MODELING_REPORT.md) | 物理模型、推断和验证设计 |
| [code/config.py](code/config.py) | 题面数据、工作参数和共同类型 |
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

配置文件仅使用 Python 标准库。输出 `Config check passed` 表示配置检查通过，不代表已完成球路预测。后续数值模块计划使用 NumPy、SciPy 和 Matplotlib。
