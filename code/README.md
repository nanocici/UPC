# 三人共用的代码起点

当前只有 `config.py` 已实现。它提供题目数据、工作参数和接口格式；其他六个 Python 文件仍由三人并行编写。

## 先检查配置

从项目根目录执行。该文件仅使用 Python 标准库，不需要先安装 NumPy 或 SciPy。

Mac（使用项目环境）：

```bash
.venv/bin/python code/config.py
```

Windows PowerShell（使用项目环境）：

```powershell
.\.venv\Scripts\python.exe code\config.py
```

尚未建立项目环境时，可分别用 `python3 code/config.py` 或 `py -3 code/config.py`。出现 `Config check passed` 表示配置检查通过，后面会列出设置；这不表示已经预测球路。

## 如何读取共同设置

其余 `.py` 文件放在同一 `code` 文件夹，使用：

```python
from config import CASE, ENVIRONMENT, SOLVER, INFERENCE

# CASE.positions_m[i] 与 CASE.times_s[i] 配对。
print(CASE.initial_position_m)
print(CASE.times_s)
print(CASE.positions_m)
print(INFERENCE.fixed_drag_coefficient)
```

统一用米、秒、千克；位置和风按 `(x, y, z)`，状态按 `(x, y, z, vx, vy, vz)`。配置对象禁止原地改写。压力情景可使用 `dataclasses.replace` 创建副本，例如 `replace(ENVIRONMENT, wind_m_s=(0.0, 1.0, 0.0))`，并明确传给自己的函数。

## 甲乙先按这个接口对接

甲在未来的 `physics.py` 中实现以下函数。这里仅约定接口，不代表函数已存在：

```python
def simulate(model, params, times_s, case=CASE,
             environment=ENVIRONMENT, solver=SOLVER):
    ...
```

- 首轮支持 `model="F"` 和 `model="S"`，`params` 按 `DynamicsParameters` 字典格式传入。
- `params` 包含 `initial_velocity_m_s`、`drag_coefficient`、`b_y`、`b_z`。S 中 `b_y` 必须为零，F 中有效向量为 `(0, b_y, b_z)`。检查参数应由求解器/拟合器实现，类型声明本身不会检查运行值。
- `times_s` 是需要返回位置的时刻，采用相对踢球起始时刻的时间坐标；本题起始时刻为零，按递增顺序传入。
- 返回符合 `SimulationResult` 的字典：`times_s`、`positions_m`、`event`、`event_time_s`、`event_position_m`、`message`。
- 积分不只覆盖两帧，还要寻找首个门线或接地事件。首个事件后不再按自由飞行外推，未取得的请求位置填 `None`，保留与请求时刻的一一对应。
- `event="goal"` 只表示首次正向穿过门线平面，可能在门框外。是否入框再按题面严格开区间判断，不能把过平面直接当成进球。
- `ground` 表示先接地；`boundary` 表示事件在数值容差内重合；`unresolved` 表示时间窗内无已确认事件；`failure` 表示求解失败，原因写入 `message`。
- 后续 G/A 基线可使用专门函数，但对外尽量返回同一结果格式。H 由丙独立实现，真实参数和完整球路不交给预测器。

## 文件归属与状态

| 人员 | 文件 | 当前状态 |
|---|---|---|
| 甲维护，三人确认 | config.py | 已创建，设置仍须在后续开发中核验 |
| 甲 | physics.py、run.py | 未实现 |
| 乙 | inverse.py、uncertainty.py | 未实现 |
| 丙 | data.py、evaluate.py | 未实现 |

乙可立即按上述接口编写拟合框架；实际拟合等待甲的运动程序。丙可同时独立编写数据生成。配置需要修改时由一人维护并通知其他人，避免三份默认值产生差异。
