# UPC 2026 Problem B：罚点球轨迹预测

根据罚球后 0.05 s 和 0.10 s 的三维位置，预测球心穿过门线的位置及不确定性。

## 当前进度

最初规划的七个 Python 文件已实现，包括统一入口 `code/run.py`。已完成程序检查与题目算例流程试跑；批量区域验证、阻力敏感性、校准和独立测试仍有待完成，现有开发结果尚未定为论文最终结果。

| 文件 | 内容 |
|---|---|
| [plan.md](plan.md) | 实施计划和模型分工 |
| [todo.md](todo.md) | 当前进度与后续任务 |
| [建模报告](reports/ANALYSIS_MODELING_REPORT.md) | 物理模型、推断和验证设计 |
| [code/config.py](code/config.py) | 题面数据、工作参数和共同类型 |
| [code/run.py](code/run.py) | 最小统一入口，默认仅运行题目 F 算例 |
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
