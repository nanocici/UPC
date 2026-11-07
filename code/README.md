# 三人共用的代码起点

当前已实现 `config.py` 和 `physics.py`。前者提供题目数据与工作参数，后者提供 S/F 正向求解和 G/A 基线。反演、不确定性、数据生成、评价及总入口仍待实现。

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

`physics.py` 现已实现以下函数：

```python
def simulate(model, params, times_s, case=CASE,
             environment=ENVIRONMENT, solver=SOLVER):
    ...
```

- 主要支持 `model="F"` 和 `model="S"`，`params` 按 `DynamicsParameters` 字典格式传入；也支持给定初速度的 G，其余气动力字段在 G 中不使用。
- `params` 包含 `initial_velocity_m_s`、`drag_coefficient`、`b_y`、`b_z`。S 中 `b_y` 必须为零，F 中有效向量为 `(0, b_y, b_z)`。检查参数应由求解器/拟合器实现，类型声明本身不会检查运行值。
- `times_s` 是需要返回位置的时刻，采用相对踢球起始时刻的时间坐标；本题起始时刻为零，按递增顺序传入。
- 返回符合 `SimulationResult` 的字典：`times_s`、`positions_m`、`event`、`event_time_s`、`event_position_m`、`message`。
- 积分不只覆盖两帧，还要寻找首个门线或接地事件。首个事件后不再按自由飞行外推，未取得的请求位置填 `None`，保留与请求时刻的一一对应。
- `event="goal"` 只表示首次正向穿过门线平面，可能在门框外。是否入框再按题面严格开区间判断，不能把过平面直接当成进球。
- `ground` 表示先接地；`boundary` 表示事件在数值容差内重合；`unresolved` 表示时间窗内无已确认事件；`failure` 表示求解失败，原因写入 `message`。
- G/A 观测拟合基线使用 `predict_baseline("G")` 和 `predict_baseline("A")`，返回同一格式。H 由丙独立实现，真实参数和完整球路不交给预测器。

## 安装并运行运动程序

在项目根目录运行。`requirements.txt` 目前列出 NumPy 和 SciPy，后续画图模块接入时再加入 Matplotlib。创建环境的步骤仅需执行一次；已有 `.venv` 时直接安装依赖。

Mac：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/physics.py
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\physics.py
```

默认运行 F 演示，初速度和横向力是人为指定值，输出不是题目的反演答案。输出中的 `positions_m` 是计算出的请求时刻位置，`event_time_s` 和 `event_position_m` 是首个事件的时间、位置，`inside_goal_opening` 仅在确认过线时判断球心是否在门框内。

命令后加 `--model S` 可运行 S 演示；加 `--model G` 或 `--model A` 可运行从题面两帧拟合的基线。例如 Mac 使用 `.venv/bin/python code/physics.py --model G`，Windows 使用 `.\.venv\Scripts\python.exe code\physics.py --model G`。

乙在 `inverse.py` 中可以直接使用：

```python
from config import CASE, INFERENCE
from physics import simulate

# 仅演示接口；真实候选参数由 inverse.py 的优化器提供。
candidate = {
    "initial_velocity_m_s": (30.0, 9.0, 7.0),
    "drag_coefficient": INFERENCE.fixed_drag_coefficient,
    "b_y": 0.0,
    "b_z": -0.15,
}
result = simulate("S", candidate, CASE.times_s)
```

读取结果后，只有在所有观测时刻的位置均非 `None`、且数值状态有效时才计算观测残差。观测窗内已接地的候选应按观测相容性排除；观测窗之后先接地的候选仍有有效观测似然，不能因没有直接门线交点而静默删除。数值失败须记录或重试，不能伪造零残差。

`simulate` 的不合法输入抛出 `ValueError`/`KeyError`，计算过程的数值异常返回 `event="failure"`。研究先验的参数范围由拟合器负责，前向程序允许范围外的有限参数用于压力测试。

运行数值检查：Mac 用 `.venv/bin/python -m unittest discover -s tests -v`；Windows 用 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`。检查覆盖解析极限、功平衡、镜像、事件顺序、近同时事件、收敛及失败状态；它们不代表已验证反演精度。`SOLVER.event_time_tolerance_s` 控制事件近同时判定，属于数值设置，仍需随边界案例检查。

## 文件归属与状态

| 人员 | 文件 | 当前状态 |
|---|---|---|
| 甲维护，三人确认 | config.py | 已创建，设置仍须在后续开发中核验 |
| 甲 | physics.py | 已实现，物理与事件检查已通过；预测能力仍需后续测试 |
| 甲 | run.py | 未实现 |
| 乙 | inverse.py、uncertainty.py | 未实现 |
| 丙 | data.py、evaluate.py | 未实现 |

乙可立即调用现有运动接口编写反演；丙可同时独立编写数据生成。配置需要修改时由一人维护并通知其他人，避免三份默认值产生差异。
