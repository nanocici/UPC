# 三人共用的代码起点

最初规划的七个文件（含 `run.py`）均已实现，包括固定阻力反演、条件后验与最小统一入口。参数依据、阻力剖面/积分、批量区域验证、校准和独立测试仍有待办。

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

在项目根目录运行。`requirements.txt` 列出 NumPy、SciPy 和可用于评价图的 Matplotlib。创建环境的步骤仅需执行一次；已有 `.venv` 时直接安装依赖。

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
| 甲 | run.py | 已实现最小入口；已运行题目 F 流程检查，开发批量入口本轮未执行 |
| 乙 | inverse.py | 已实现固定阻力 S/F 的 MAP/LS、多起点、诊断和预测导出 |
| 乙 | uncertainty.py | 已实现固定 Cd 带权条件后验、事件概率、均值/格网众数、区域与采样诊断 |
| 丙 | data.py | 已实现独立 H 生成、误差、数据分组、文件读写及带单位 CSV 读取 |
| 丙 | evaluate.py | 已实现评价、分层、整组重采样、预测格式和 G/A 开发集示例；S/F 与概率区域待接入 |

反演和条件后验输出均可交给评价器；后续优先完成阻力剖面与区域检验。配置需要修改时由一人维护并通知其他人，避免三份默认值产生差异。

## 使用 data.py 生成测试数据

`data.py` 不导入 `physics.py` 或 `inverse.py`，可以单独运行。它独立计算真值球路，然后提取前两帧并加入测量误差。五类生成情景为常系数、变阻力、自旋相关横向力、风、组合机制；气动曲线和随机参数属于模拟压力设置，尚未匹配具体球型的实验数据。

Mac：

```bash
.venv/bin/python code/data.py --sources 60 --output data/my_first_dataset
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\data.py --sources 60 --output data\my_first_dataset
```

`--sources 60` 是 60 条独立源射门。默认每条射门对应 4 个噪声档位，因此有 240 条观测记录，独立样本数仍是 60。默认按源射门 60%/20%/20% 分为 36 条开发、12 条校准、12 条测试射门；同源噪声副本始终留在同一组。这个规模用于接通程序，不足以单独证明区域覆盖率已可靠校准。较大实验可指定 `--sources 300`，仍须考虑各机制分层样本数。

输出目录结构如下，其中每个 `.jsonl` 文件每行是一条 JSON 记录：

```text
my_first_dataset/
  manifest.json                 生成设置、种子、样本数量与事件统计
  observations/
    development.jsonl           开发输入
    calibration.jsonl           校准输入
    test.jsonl                  独立测试输入
  truth/
    development.jsonl           开发真值与生成参数
    calibration.jsonl           校准真值与生成参数
    test.jsonl                  测试真值与生成参数
```

乙仅加载 `observations` 文件。`truth` 和 `manifest` 中的种子、真实球路和参数供丙复现、分层及判卷，不能传给预测器。预测输入不含真实气动参数、生成机制标签、真实落点或生成种子。默认风对预测器未知：真实风只在真值中，输入中的名义风为零并标明 `wind_known=false`；另加 `--wind-known` 才显式提供实际风速。

输出目录已存在时程序会拒绝覆盖。重新生成应使用另一个 `--output` 路径；同样的种子、设置及数值环境可复现相同内容。当前已生成的示例位于 `data/synthetic_demo`，运行上述命令会另建一个目录。

乙可以这样取得共同 `ObservationCase`（尚不调用未实现的反演）：

```python
from config import DATA_DIR, Environment
from data import load_observations, observation_case

records = load_observations(DATA_DIR / "synthetic_demo" / "observations" / "development.jsonl")
record = next(row for row in records if row["observation_available"])
case = observation_case(record)
known = dict(record["known_environment"])
known["wind_m_s"] = tuple(known["wind_m_s"])
environment = Environment(**known)
# 将 case 和 environment 显式传给之后的反演接口。
```

这段示例只挑一条可用记录展示接口。正式批量流程必须统计所有记录，包括先接地、缺少观测或生成失败的射门；`observation_available=false` 的记录不能静默消失。`observation_case` 会拒绝把这种记录转成完整两帧输入，`load_observations` 会拒绝真值文件。真值文件一条对应一条源射门，评价时用 `source_id` 与多条观测副本关联。

其他误差实验：`--noise-mode rounding` 仅作厘米舍入，`--noise-mode none` 不加误差，二者每个源射门只生成一个副本；高斯模式加 `--correlation 0.5` 表示同一坐标跨帧的等相关误差。误差不会被裁剪到地面或球门内。

实测 CSV 接入使用 `load_trajectory_csv(path, length_unit="dm", time_unit="frame", fps=200)`；文件须有 `time,x,y,z` 列，函数会转成 m 和 s。此例的单位与 FPS 只是调用示范，实际值必须核查。函数不自动识别起飞时刻、变换球门坐标或确保早期观测未使用后段信息，外部数据适用性仍待独立核验。

## 使用 evaluate.py 判卷

评价器可以先运行，不依赖尚未实现的 `inverse.py`。默认只用 G/A 基线在 `synthetic_demo` 的开发组接通流程，先由公开观测生成预测，然后才读真值。`--plot` 另存 PNG；输出目录存在时拒绝覆盖，应另取名字。

Mac：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/evaluate.py --output results/my_evaluation --plot
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\evaluate.py --output results\my_evaluation --plot
```

默认数据路径是 `data/synthetic_demo`，可用 `--dataset` 改变。输出包括 `summary.json`（方法汇总、分层、配对比较和指标定义）、`metrics.csv`（简表）、`per_case.csv`（逐案例记录）、`predictions.jsonl`（本次预测）以及可选的 `point_metrics.png`。CSV 使用带 BOM 的 UTF-8，方便 Windows Excel 打开。已有示例位于 `results/evaluation_demo`。

可用 `--bootstrap 0` 暂时省略区间；默认 300 次整组重采样仅为流程起步设置。区间按 `source_id` 重采样，同源噪声副本始终一起抽取。分层表分别按观测标准差和生成机制汇总，机制只在评价端从真值读取。样本少时不能把区间作为充分验证。

## 乙需要交给评价器的预测格式

每个方法、每个 `case_id` 写一行 JSON。字段固定如下（示例为接口展示，数值不是反演结果）：

```json
{
  "schema_version": 1,
  "case_id": "source_00000_noise_00",
  "source_id": "source_00000",
  "split": "development",
  "method": "F_fixed",
  "status": "ok",
  "event": "goal",
  "point_kind": "point_estimate",
  "point_yz_m": [2.0, 1.5],
  "arrival_time_s": 0.4,
  "region_mass": null,
  "region": null,
  "goal_probability": null,
  "message": ""
}
```

- `case_id`、`source_id`、`split` 必须原样取自观测；`method` 明确模型与推断版本，同方法不得重复预测同一案例。
- `status` 用 `ok`、`failure`、`unavailable`。缺少提交时评价器记为 `missing`，所有这些状态均保留在逐案例表中。失败/不可用时点、时间、区域和概率应为 `null`。
- `event` 为共同事件名或 `null`。先接地、边界和未决属于可报告物理结果，不自动算数值求解失败。门线评分点、到达时间有定义时才提供，不能拿接地点代替门线交点。
- `point_kind` 写明 `point_estimate`、`mean` 或 `mode` 等实际含义；单条 MAP 轨迹交点不自动是门线均值/众数。同方法全部记录的定义保持一致，换定义应另起方法名。
- `region_mass` 表示方法声明的名义区域概率，如 0.90。方法声称提供区域时，失败记录也保留这个值，才能将漏预测计入覆盖率。只提供点预测的基线用 `null`，此时覆盖率未定义。
- 区域使用 `region={"center_yz_m":[...],"covariance_m2":[[...],[...]],"threshold":...}`，表示 `(g-center)' covariance^-1 (g-center) <= threshold`。`threshold` 是平方马氏距离阈值，已校准时传最终阈值；面积为 `pi*threshold*sqrt(det(covariance))`。本模块只评价区域，不拟合校准尺度。
- `goal_probability` 为接地前直接进入题面球心开口的概率。无概率方法用 `null`；不能把一个拟合点入框与否伪装成经过校准的概率。

`point_prediction(observation, method, simulation, point_kind="point_estimate")` 可将 `physics.py` 的单条结果包装成该格式。拟合参数、优化日志另存，不直接混入这个预测对象。后续 `uncertainty.py` 可填入均值、众数、区域和事件概率。

读取已提交预测进行评价，例如 Mac：

```bash
.venv/bin/python code/evaluate.py --predictions results/my_predictions.jsonl --output results/my_scores --plot
```

Windows 使用 `.\.venv\Scripts\python.exe code\evaluate.py --predictions results\my_predictions.jsonl --output results\my_scores --plot`。`--predictions` 可接多个文件，合并后按方法评价；相同方法相同案例重复时拒绝。默认评价开发组；校准/测试组需显式指定 `--split calibration` 或 `--split test`，并提供对应组的预测。方法选择、校准尺度与配置冻结需在研究流程中另行记录，不根据独立测试结果回选模型。

## 评价结果怎样解读

- **二维误差和时间误差**：只在真值直接过门线且预测量有定义的案例计算，始终附带有效案例数。两种方法比较另给共同可评分案例上的配对结果，避免只比较各自成功子集的中位数。
- **覆盖率**：对声称提供概率区域的方法，以所有真实直接过线案例为分母，漏交、失败、缺区域、奇异/无效区域均计漏覆盖；另外报告观测完整子集的覆盖率。没有概率区域的基线输出 `null`，并不伪造覆盖率。
- **面积**：主面积分布仅统计真实直接过线案例中有效区域的面积。区域提供数、覆盖率与失败情况一起看，不能仅因区域变大就声称方法改善。
- **Brier 分数**：在真值直接入框/越框、先接地等标签明确案例计算；边界、未决和生成失败排除并保留计数。缺概率的案例另报数量，不自动填 0。
- **样本量**：区分噪声观测案例数和独立源射门数。`failure_or_missing_rate` 不含 `unavailable`，后者单列；真值直接过线中的点预测提供比例同时体现没有门线输出的情况。

G/A 开发示例与 S/F 点预测已接入。条件后验的均值、区域与事件概率接口已实现，实际区域覆盖率、区域校准和独立测试仍待批量检验。

## 使用 inverse.py 反推参数

本版固定阻力（默认 `INFERENCE.fixed_drag_coefficient=0.22`），S 拟合 4 个参数，F 拟合 5 个参数；不从两帧额外估计风、质量或阻力曲线。优化时每次调用 `physics.simulate`，只积分到最后观测时刻，找到参数后再算完整首个事件。

先运行题目算例。Mac：

```bash
.venv/bin/python code/inverse.py --output results/my_inverse_example
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\inverse.py --output results\my_inverse_example
```

默认同时算 S/F 的 MAP。输出 `example.json`、`fits.jsonl` 和 `run_config.json`；已有本轮示例在 `results/inverse_example`。结果是工作误差、先验和固定阻力条件下的参数点估计，尚不是门线最可能点或可信区域。

命令选项：`--model S` 或 `--model F` 单独运行；`--mode ls` 明确去掉先验罚项；`--drag 0.30` 改变固定阻力；`--sigma 0.02` 显式改变高斯误差假设；`--starts 3` 指定 1–9 个默认起点。每次更换输出目录，避免覆盖。

默认 MAP 目标为 `0.5*e.T*Sigma^-1*e + prior_penalty`。F 的横向先验为半径 0.50 圆盘内、尺度 0.20 的二维各向同性高斯；S 使用它的 b_z 边缘先验，罚项包含沿 b_y 积分带来的 erf 因子。常数选择使各自零点罚项为零，它们不用于比较模型证据。LS 保留相同物理参数范围，只去掉正则项。报告同时保存数据卡方、先验罚项、总目标与米制残差，不能把 LS 数字写成 MAP 结果。

拟合采用缩放参数的多起点 SLSQP，F 显式满足横向圆盘约束。只接收已收敛且观测窗相容的候选；数值失败或早于观测接地不会伪造残差。第二帧后接地仍是可接受拟合，但没有直接门线点。最终传播数值失败与观测窗拟合成功分别记录。

拟合日志保存每个起点状态、参数边界命中、缩放后纯数据雅可比及其秩/奇异值/条件数。先验没有被混入数据雅可比；局部数值满秩不证明参数已被精确识别，也不证明全局唯一。此版不输出参数协方差或概率区域。

## 用反演处理合成观测并交给评价器

Mac：

```bash
.venv/bin/python code/inverse.py --dataset data/synthetic_demo --split development --output results/my_inverse_development
.venv/bin/python code/evaluate.py --predictions results/my_inverse_development/predictions.jsonl --output results/my_inverse_scores --plot
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\inverse.py --dataset data\synthetic_demo --split development --output results\my_inverse_development
.\.venv\Scripts\python.exe code\evaluate.py --predictions results\my_inverse_development\predictions.jsonl --output results\my_inverse_scores --plot
```

反演只打开 `observations`，真实参数与 `truth` 不参与初值、先验或优化。批量保存 `predictions.jsonl`、`fits.jsonl` 和 `run_config.json`，每个模型、每条观测保留一条记录；每 10 条观测打印进度。方法名如 `F_cd0.22_map`，评分点标签为 `map_trajectory` 或 `ls_trajectory`，均不冒称门线均值或众数。

批量高斯误差采用记录提供的总标准差和跨帧相关系数，协方差按 `[x1,y1,z1,x2,y2,z2,...]` 排列。API 也支持显式正定协方差。纯舍入、无噪声及完全相关的奇异误差模型尚未实现硬约束似然；默认将此类案例标为失败并记录原因。使用 `--sigma` 是显式引入高斯工作假设，不是替它们实现了精确似然。

单案例可直接调用：

```python
from inverse import fit_case
fit = fit_case("F", mode="map", starts=3)
if fit["success"]:
    print(fit["parameters"])
    print(fit["rms_residual_m"])
    print(fit["simulation"])
```

`fit_case` 支持 `case`、`environment`、`solver`、`settings`、`drag_coefficient`、`sigma_m`、`frame_correlation`、`covariance_m2` 与 `initial_guesses`；自定义初值按共同参数顺序提供。之后的阻力剖面可改变 `drag_coefficient` 并每层重拟合，当前尚未批量实现剖面或阻力积分。

## 使用 uncertainty.py 计算预测范围

先运行题目单算例，不必先批量跑整个数据集。默认 F，也可 `--model S` 或 `--model both`。默认 `--samples 2000` 表示基础 N，实际抽取 2N=4000 次提议，用前 N 与全部 2N 比较稳定性。计算量大于单次反演；具体耗时以本机日志为准。

Mac：

```bash
.venv/bin/python code/uncertainty.py --model both --output results/my_uncertainty_example --plot
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\uncertainty.py --model both --output results\my_uncertainty_example --plot
```

每处理 500 个传播候选会打印进度。输出包括 `summary.json`（拟合、事件概率、位置/区域和诊断）、`run_config.json`、每案例 `candidates_编号.csv`（全部提议参数、权重、事件及密度），以及可选的分布 PNG。本轮题目示例已保存至 `results/uncertainty_example`。已有目录不会覆盖，另取输出名即可。

这些是**固定阻力、给定环境与先验下的条件概率**。未包含阻力未知、模型形式和环境频率的不确定性，模型内 90% 区域尚未校准为实测 90% 覆盖保证。

## 采样方法和诊断

1. 先调用固定 Cd MAP，保持 inverse.py 的误差协方差和 S/F 匹配先验。
2. 在缩放参数 q 坐标中，对包含先验的完整负对数后验求数值曲率。曲率不正定或不可计算时，尝试 `J_data.T @ J_data + H_prior`，且明确仅作为提议形状；仍不可用则失败，不伪造协方差。
3. 建立局部与宽尾的 Student t 混合提议，多起点发现的不同高权重区域也可加入。默认 df=5、局部/宽分量质量 0.85/0.15、宽分量标准差放大 3 倍。t 的 shape 与协方差已按 `(df-2)/df` 转换。
4. 对全部固定次数提议计算真实似然与先验，以 `log_weight = log_target - log_mixture_proposal` 修正采样密度。同一固定 Cd 模型中，参数缩放雅可比是共同常数，可在归一化时约去；这里不计算跨模型证据。
5. 参数域外或与观测窗物理事件不相容时权重为零。观测窗数值失败先收紧设置重试，仍失败则停止发布概率。似然已知后传播失败的权重保留，归入未判定事件。
6. 默认要求总体 ESS 至少 300、条件位置 ESS 至少 100。N→2N 的最大事件概率变化不超过 0.02、条件均值移动不超过 0.05 m、区域面积相对变化不超过 15%，且未判定事件质量不超过 1e-6。未通过时结果保留诊断和候选，批量预测标为失败，避免把不稳定区域计为有效。

总体采样通过不自动证明最可能位置稳定。众数使用与门框对齐的等面积格网，另检查半格宽、半格平移和 N→2N 的众数格变化。当前 0.20 m 的众数移动阈值是工作精度档位；格中心不是无限分辨率的精确峰值，也没有预设双峰。`local_gaussian_basic_checks_passed` 只表示曲率/参数边界的基础检查，不证明后验在全局为高斯；最终分布始终由真实权重获得。

`direct_crossing.mode_stable` 和 `inside_conditioned.mode_stable` 另行标记对应众数是否通过格网、样本量及条件 ESS 检查。`success=true` 表示均值、区域和事件概率的采样诊断通过，不保证众数稳定。若众数仍不稳定，应增加 `--samples` 并复核格网；不能把通过均值检查当成最可能点的精确保证。

## 怎样读取概率与区域

- `summary.event_probabilities` 包含直接入框、直接越框、先接地、边界、未决和传播失败，质量保留在同一个分母中。
- `direct_inside_probability_bounds` 为已确认直接入框质量，加上未判定质量后的上下界。未判定质量非零时，批量导出的单一 `goal_probability` 留空，保留界限供研究解释。
- `direct_crossing` 条件于已确认直接过平面的候选；`inside_conditioned` 再条件于入框。两者分别报告均值、格网众数、条件 ESS、时间和区域。
- 区域以加权均值、加权总体协方差和平方马氏距离的加权 90% 分位数形成椭圆。它是模型内区域；多峰可能使椭圆包含低密度空白，应结合候选图看。
- 批量评分点固定使用条件均值 `point_kind="mean"`，到达时间使用对应条件中位数；格网众数单独保留。没有足够条件 ESS 时不发布该位置。没有直接过线候选时不伪造点或区域。

批量接口示例（计算成本较高，建议先单算例计时；这里只会读取指定组观测）：

```bash
.venv/bin/python code/uncertainty.py --dataset data/synthetic_demo --split development --model F --output results/my_uncertainty_development
.venv/bin/python code/evaluate.py --predictions results/my_uncertainty_development/predictions.jsonl --output results/my_uncertainty_scores --plot
```

Windows 将前缀替换为 `.\.venv\Scripts\python.exe`，路径可使用反斜杠。`--plot` 分布图用于题目单算例，批量比较图由 evaluate.py 产生。诊断失败和观测不可用的记录仍导出相同均值标签与名义区域质量，以便评价器计入失败和漏覆盖。

API：`sample_case("F", ...)` 返回完整条件后验结果，`uncertainty_observation(record, "F", ...)` 返回 `(共同预测记录, 完整日志)`。`UncertaintySettings` 提供样本数和诊断阈值，其他参数沿用共同 case、environment、solver、settings、观测协方差、固定 Cd 和随机种子。更改阈值必须留存配置，不能为了通过诊断临时放宽正式阈值。

## 最小统一入口 run.py

当前只完成“统一调用与记录”这一步。运行题目算例：

Mac：

```bash
.venv/bin/python code/run.py --plot
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe code\run.py --plot
```

默认只运行 F，调用 inverse.py，再用相同设置调用 uncertainty.py。后者内部仍按自己的接口重新做 MAP，没有新增算法或改用前一文件作为后验。题目没有真实门线答案，因此单算例不计算定位误差，也不调用评价器冒充验证。

默认在 results 下自动建立带 Asia/Shanghai 时间的唯一目录；也可加 `--output results/my_run` 指定新目录，已有目录不会覆盖。终端显示每个阶段开始/完成，详细进度与标准输出、错误输出都写入日志。

```text
results/run_example_时间/
  run_config.json       共同设置、命令、Python/包版本、源文件 SHA256
  run_summary.json      执行状态、耗时和已有诊断提示
  logs/
    01_inverse.log
    02_uncertainty.log
  inverse/             原 inverse.py 的输出
  uncertainty/         原 uncertainty.py 的输出
```

可选 `--model S` 或 `--model both`；`--samples`、`--starts`、`--seed`、`--drag`、`--sigma` 转发给适用模块。默认 N=2000（实际 4000 次提议）与原程序相同。脚本用当前 `sys.executable` 启动子模块，使用参数列表传递，兼容带空格路径；依赖不足时先安装 requirements.txt。

进程失败则停止后续阶段，保留日志，并以非零退出码结束。进程全部执行完成时 `status=completed`；科学诊断问题另外保存在 `review_items` 与 `diagnostics_status=review_required` 中，例如 F 众数不稳定。执行完成不代表模型已通过外部验证，也不会自动调大样本数或放宽阈值。

第二种方式为 `--task development --dataset data/synthetic_demo`，仅编排已有的 S/F 点反演、G/A 基线和开发组评价，默认仍只选 F；需要 S/F 时加 `--model both`。这一最小入口不会自动生成数据，也不自动访问校准/测试组。批量概率区域、阻力剖面、敏感性、校准和独立测试不在本次入口任务中。本轮仅实际执行了题目 F 算例；开发模式只核对调用参数与边界。

已保存本轮流程检查于 `results/run_example_check`，说明见 reports/RUN_CHECK_REPORT.md。后续安排实验前，先根据 plan.md 和 todo.md 确定研究设置与预算。
