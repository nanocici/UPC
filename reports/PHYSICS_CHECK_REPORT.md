# physics.py 实现与检查记录

日期：2026-11-07（Asia/Shanghai）。本轮从共享配置开始实现运动程序，不沿用历史拟合结果。

## 已实现的接口

- `simulate(model, params, times_s, case, environment, solver)`：F/S 正向求解，以及给定初速度的 G。
- `gravity_initial_velocity`：固定起点与重力的同方差位置最小二乘初速度。
- `constant_acceleration_parameters`：固定起点的恒加速度最小二乘参数。
- `predict_baseline("G"/"A")`：拟合简单基线并按共同事件规则传播。
- `acceleration` 与 `is_in_goal_opening`：加速度诊断及题面球心开口判定。

F/S 使用相对气流速度和配置中的质量、半径、重力及空气密度。阻力反向，有效横向项为 `(0,b_y,b_z) × u`，S 严格要求 `b_y=0`。研究先验边界不强加在前向求解器中。

积分使用 SciPy `solve_ivp` 的连续插值和事件求根，默认 RK45。初始时间窗未发现事件则延长至最大时间窗，持续保留首个事件前的观测采样。门线只接受正向穿越；地面只接受下降穿越；初始离地零点通过消去已知时间因子处理。首个事件之后的位置不外推，保留为 `None`。近同时过线与接地标为 `boundary`。过平面和进入门框分开判断。

近同时判定采用法向距离、当前法向速度和配置的 `event_time_tolerance_s`，附加位置舍入容差；这是数值边界标记，不是精确碰撞模型。事件查找采用逐步符号变化，最大步长和边界情景仍需在扩大应用范围时复核。[SciPy solve_ivp 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

## 本轮检查

执行环境：macOS，Python 3.13.13、NumPy 2.5.4、SciPy 1.18.1。Windows 命令已提供，但本轮未在 Windows 实机运行。

`tests/test_physics.py` 的 21 项检查通过，覆盖：

| 类别 | 检查与支持范围 |
|---|---|
| 方程和极限 | 量纲核对；无空气力对照解析重力解；零重力/空气力匀速解；纯二次阻力对照解析解。 |
| 力学关系 | 静风和恒风下横向力与相对速度垂直；瞬时功平衡；纯阻力水平航向保持。 |
| 结构与对称 | S 与受限 F 完全一致；含横风情景的横向镜像对称。 |
| 数值收敛 | 步长减半且位置/速度绝对容差、相对容差收紧十倍后，演示 F 案例交点变化小于 0.005 m，事件时间变化小于 0.001 s。 |
| 初始与事件 | 离地不误判起点为落地；短跳；起始接触；先接地；门外过线；开口严格边界；同时及近同时事件；起始过线。 |
| 时间与失效 | 延长积分窗口；无事件未决；非零起始时间；无效输入拒绝；显式数值异常状态；首个事件之后不返回自由飞行位置。 |
| 基线 | A 复原两帧并取首次正向过线根；G 的到达时间与解析时间相符。 |

另外，F/S/G/A 四种命令行示例均成功输出 JSON，更新后的共享配置检查通过。

## 如何复现

从项目根目录运行，依赖见 requirements.txt。

Mac：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python code/physics.py
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe code\physics.py
```

## 当前证据边界

这些检查支持本轮方程实现、事件处理和演示情景下的数值一致性。默认 F/S 命令行案例使用人为指定参数，未进行两帧气动力反演。G/A 为简单拟合基线。尚未完成参数先验核验、不确定性传播、区域校准、异机制模拟和独立实测评价，因此没有据此声明题目落点预测精度或入框概率。
