"""Build paper-facing synthesis from saved evidence only; never samples or fits."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,text):
    (OUT/name).write_text(text.strip()+'\n',encoding='utf-8')


def pct(x):return f'{100*x:.2f}%'
def fmt(x):return f'{x:.4f}'


def main():
    drag=read('results/整理阻力剖面实验/summary.json')
    stability=read('results/算例稳定性实验/summary.json')
    performance=read('results/最终主点性能实验/summary.json')
    release=read('results/最终主点性能实验/stability_summary.json')
    case=read('results/final_case/final_case.json')
    previous=read('results/independent_200/independent_summary.json')
    numerical=read('results/最终主点性能实验/numerical_validation.json')
    validation=read('results/最终主点性能实验/validation.json')
    main_stats=performance['primary_sigma_001']; all_stats=performance['overall']
    ci=performance['primary_source_bootstrap']['paired_median_difference_f1_minus_g_ci95_m']
    all_ci=performance['source_bootstrap']['paired_median_difference_f1_minus_g_ci95_m']
    improvement=1-main_stats['paired_peak']['median_m']/main_stats['paired_g']['median_m']
    assert main_stats['true_inside_sources']==49 and main_stats['stable_inside_count']==49
    assert all_stats['true_inside_sources']==49 and all_stats['true_inside_observations']==196
    assert drag['verification_count']==drag['verification_pass_count']==20
    assert stability['all_diagnostics_pass'] and validation['all_weight_checks_pass']
    assert numerical['all_pass'] and validation['prediction_seal_unchanged_after_scoring']
    sources=[
        'results/整理阻力剖面实验/summary.json',
        'results/整理阻力剖面实验/profile.csv',
        'results/整理阻力剖面实验/validation.json',
        'results/算例稳定性实验/summary.json',
        'results/算例稳定性实验/kde_peaks.csv',
        'results/算例稳定性实验/validation.json',
        'results/最终主点性能实验/summary.json',
        'results/最终主点性能实验/per_case.csv',
        'results/最终主点性能实验/stability.csv',
        'results/最终主点性能实验/stability_summary.json',
        'results/最终主点性能实验/prediction_manifest.json',
        'results/最终主点性能实验/numerical_validation.json',
        'results/最终主点性能实验/validation.json',
        'results/final_case/final_case.json',
        'results/independent_200/independent_summary.json']
    figures=[
        ('主图1','fig01_drag_fit.png','results/整理阻力剖面实验/拟合剖面图.png','阻力重拟合剖面',
         '指定算例的F/S纯数据LS剖面、F0/F1/S的MAP目标及其分解。竖线为冻结Cd=0.22；网格最小层不是实测阻力。MAP目标不能跨模型直接作证据排名。'),
        ('主图2','fig02_mode_convergence.png','results/算例稳定性实验/sampling_convergence.png','指定算例平滑峰的采样量检查',
         '入框条件、带宽0.10m；每种子4k至64k前缀的峰位置，虚线为四种子合并64k峰。前缀来自同一固定提议分布，不是各预算重新构造提议的独立实验。'),
        ('主图3','fig03_point_errors.png','results/最终主点性能实验/point_errors.png','最终平滑峰与G的误差比较',
         '49条真实入框源的四档噪声中，193组共同成功观测的误差CDF与配对散点；同源副本相关。该图是四档合计，不是仅1cm层；49条1cm主结果见性能总表。'),
        ('补充图S1','figS01_region_stability.png','results/最终主点性能实验/coverage_and_stability.png','覆盖率及稳定输出分层',
         '区域覆盖以真实直接过线案例为分母；稳定输出比例以真实入框案例为分母。两条曲线样本条件不同，不能直接比较大小来判断同一任务的成功率。'),
        ('补充图S2','figS02_case_density.png','results/算例稳定性实验/kde_density.png','题目指定算例的平滑密度',
         '两种条件的0.10m合并KDE及其峰。白色线是峰值50%和80%的等密度线，不是90%预测区域；无真实后续位置，不能凭此图评价准确性。'),
        ('补充图S3','figS03_drag_goal.png','results/整理阻力剖面实验/门线预测变化图.png','阻力变化对门线量的影响',
         '实线逐层重拟合，虚线固定名义MAP其他参数的前向扫描。轨迹MAP点不是正式KDE主点，扫描跨度也不是预测误差或概率区间。'),
        ('补充图S4','figS04_bandwidth.png','results/算例稳定性实验/bandwidth_sensitivity.png','指定算例带宽敏感性',
         '0.05/0.10/0.20m带宽下各64k种子与等权合并峰。支持指定算例对所检验平滑尺度较稳定，不支持任意带宽或无限分辨率唯一众数。')]
    (OUT/'figures').mkdir(exist_ok=True)
    for _,name,source,_,_ in figures:shutil.copy2(ROOT/source,OUT/'figures'/name)

    # All counts below are rederived from sealed predictions and the saved score table.
    import csv
    scored=list(csv.DictReader((ROOT/'results/最终主点性能实验/per_case.csv').open(encoding='utf-8-sig')))
    predictions={r['case_id']:r for r in (json.loads(s) for s in (ROOT/'results/最终主点性能实验/predictions.jsonl').read_text().splitlines())}
    eligible=[r for r in scored if r['truth_inside']=='True']
    primary_rows=[r for r in eligible if float(r['sigma_m'])==.01]
    raw_all=sum(predictions[r['case_id']]['conditions'].get('inside_conditioned',{}).get('raw_grid',{}).get('pass_',False) for r in eligible)
    raw_main=sum(predictions[r['case_id']]['conditions'].get('inside_conditioned',{}).get('raw_grid',{}).get('pass_',False) for r in primary_rows)
    covered_main=sum(r['region_covered']=='True' for r in scored if r['truth_direct']=='True' and float(r['sigma_m'])==.01)
    covered_all=sum(r['region_covered']=='True' for r in scored if r['truth_direct']=='True')
    assert (raw_main,raw_all,covered_main,covered_all)==(20,77,68,267)
    for rows_,stats in ((primary_rows,main_stats),(eligible,all_stats)):
        pair=[r for r in rows_ if r['peak_error_m'] and r['g_error_m']]
        import numpy as np
        for key,field in (('paired_peak','peak_error_m'),('paired_g','g_error_m')):
            assert np.isclose(np.median([float(r[field]) for r in pair]),stats[key]['median_m'])

    table=['# 论文性能总表','',
           '日期：2026-11-08。数值直接取自保存的实验结果，并与逐案例评分复核。m为米，m²为平方米。','',
           '## 表1：主要结果——观测标准差1cm，真实入框49条独立源射门','',
           '| 方法 | 共同成功观测数 | 中位误差/m | P90/m | RMSE/m |','|---|---:|---:|---:|---:|']
    for label,key in [('F1固定带宽平滑峰','paired_peak'),('G重力抛体基线','paired_g'),('F1条件均值，仅辅助','mean_auxiliary')]:
        a=main_stats[key];table.append(f"| {label} | {a['n']} | {fmt(a['median_m'])} | {fmt(a['p90_m'])} | {fmt(a['rmse_m'])} |")
    table+=['',f'平滑峰相对G的中位误差下降{100*improvement:.1f}%。F1−G中位误差差的源级配对bootstrap 95%区间为[{ci[0]:.4f},{ci[1]:.4f}]m。该比较不证明F1优于所有其他方法；均值列没有单独做优越性检验。','',
            '## 表2：四档噪声合计的配对点误差','',
            '| 方法 | 共同成功观测数 | 中位误差/m | P90/m | RMSE/m |','|---|---:|---:|---:|---:|']
    for label,key in [('F1固定带宽平滑峰','paired_peak'),('G，同一193组观测','paired_g')]:
        a=all_stats[key];table.append(f"| {label} | {a['n']} | {fmt(a['median_m'])} | {fmt(a['p90_m'])} | {fmt(a['rmse_m'])} |")
    table+=['',f'来自49条真实入框源的196组观测，193组进入配对表；不是193条独立射门。F1−G配对区间[{all_ci[0]:.4f},{all_ci[1]:.4f}]m。G在全部可用真实入框案例上为195组，中位误差{all_stats["g_all_eligible"]["median_m"]:.4f}m，P90 {all_stats["g_all_eligible"]["p90_m"]:.4f}m。','',
            '## 表3：稳定输出比例——必须与点误差同时报告','',
            '| 分母与条件 | 稳定平滑峰数/总数 | 比例 |','|---|---:|---:|',
            f"| 全部输入，四档噪声 | {release['stable_inside_conditioned_predictions']}/400 | {pct(release['stable_inside_conditioned_predictions']/400)} |",
            f"| 完整可用输入，四档噪声 | {release['stable_inside_conditioned_predictions']}/380 | {pct(release['stable_inside_conditioned_predictions']/380)} |",
            f"| 真实入框，四档噪声 | {all_stats['stable_inside_count']}/196 | {pct(all_stats['stable_inside_rate'])} |",
            '| 真实入框，1cm层 | 49/49 | 100.00% |','',
            '400组输入=100源×4噪声；380组可用、374组完成后验采样、6组拟合失败、20组无完整观测。真实入框子集的3组未发布均因采样诊断未通过。49/49是样本结果，不是总体100%稳定保证。','',
            '## 表4：直接过线条件区域的经验覆盖与面积','',
            '| 本次四种子合并结果 | 真实直接过线观测数 | 提供区域数 | 覆盖数 | 覆盖率 | 面积中位/m² |','|---|---:|---:|---:|---:|---:|',
            f"| 1cm层 | 77 | {main_stats['region_provided']} | {covered_main} | {pct(main_stats['region_coverage'])} | {fmt(main_stats['region_median_area_m2'])} |",
            f"| 四档合计 | 308 | {all_stats['region_provided']} | {covered_all} | {pct(all_stats['region_coverage'])} | {fmt(all_stats['region_median_area_m2'])} |",'',
            '1cm层对应77条独立过线源；四档合计仍为77源的308组观测。缺失区域计为漏覆盖；面积只对已提供区域统计。不能称严格90%覆盖保证。','',
            '## 表5：噪声分层','',
            '| sigma/m | 真实入框源数 | 稳定数/观测数 | 配对F1中位误差/m | 配对G中位误差/m | 区域覆盖率 | 面积中位/m² |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for label,a in performance['by_noise'].items():
        table.append(f"| {label} | {a['true_inside_sources']} | {a['stable_inside_count']}/{a['true_inside_observations']} | {fmt(a['paired_peak']['median_m'])} | {fmt(a['paired_g']['median_m'])} | {pct(a['region_coverage'])} | {fmt(a['region_median_area_m2'])} |")
    table+=['','## 表6：机制分层，四档合计','',
            '| 机制 | 真实入框源数 | 稳定数/观测数 | 配对F1中位误差/m | 配对G中位误差/m | 区域覆盖率 |','|---|---:|---:|---:|---:|---:|']
    for label,a in performance['by_mechanism'].items():
        table.append(f"| {label} | {a['true_inside_sources']} | {a['stable_inside_count']}/{a['true_inside_observations']} | {fmt(a['paired_peak']['median_m'])} | {fmt(a['paired_g']['median_m'])} | {pct(a['region_coverage'])} |")
    table+=['','各机制样本量不同；区域覆盖以该机制的真实直接过线案例为分母，不是表中的入框观测数。未对各机制分别证明统计显著的模型优越性。']
    write('性能总表.md','\n'.join(table))

    report=['# 三项实验总报告：当前 F1 的证据与论文口径','',
            '日期：2026-11-08（Asia/Shanghai）。本报告只汇总已完成实验，不重新采样、不调模型或校准参数。','',
            '## 结论先行','',
            f'当前证据支持：在已检验的合成射门数据中，固定带宽平滑峰可稳定计算，并在1cm噪声的真实入框子集上相较G降低中位位置误差约{100*improvement:.1f}%。不确定性区域仍欠覆盖；不主张真实球场上的精度保证、连续密度唯一众数或严格90%覆盖。',
            '三项实验各自回答阻力敏感性、指定算例的数值稳定性、批量主点的预测性能。相互补充，不能互相替代。','',
            '## 1. 冻结方案和统一术语','',
            '| 对象 | 本报告的统一含义 |','|---|---|',
            '| F1 | 重力＋二次阻力＋有效横向力；Cd=0.22；by/bz先验尺度0.10/0.20，圆盘约束半径0.50 |',
            '| F0 | 同一F方程，by/bz尺度0.20/0.20；本轮剖面对照 |',
            '| S | by=0的简化结构，沿用当前默认bz边缘先验，匹配F0；不是完全控制先验的F1结构消融 |',
            '| G | 由相同早期观测拟合的重力抛体基线；预测不裁剪到门框内 |',
            '| 正式主点 | 已入框条件下，0.10m固定带宽、0.01m求值网格的KDE平滑峰；各种子条件归一化后等权合并 |',
            '| 原始格网众数 | 格网内质量最大的格中心，独立诊断；不借用平滑峰的通过状态 |',
            '| 辅助均值 | 相应事件条件下的后验均值，不称为最可能点 |',
            '| 预测区域 | 直接过门线条件下的经验校准椭圆，冻结阈值3.08352545；与入框条件主点分别评价 |','',
            '批量实验使用每条观测记录给定的噪声水平；1cm层为主要结果，0.5/2/5cm层为分层敏感性。保持当前已知测量精度假设，不宣称已检验未知噪声错设。','',
            '## 2. 三项实验的证据链','',
            '| 实验 | 输入及对照 | 可以支持 | 不能支持 |','|---|---|---|---|',
            '| 一：阻力重拟合剖面 | 指定算例，26层Cd，F/S LS和F0/F1/S MAP | 阻力的条件敏感性、重拟合补偿、数值复核 | 真实Cd已确定、Cd=0.22最优、毫米级总精度 |',
            '| 二：算例主点稳定性 | 指定算例，4种子×64k，前缀与带宽对照 | 给定模型和平滑尺度下的可重复性 | 算例真实误差、其他射门必然稳定、唯一连续众数 |',
            '| 三：批量稳定性＋主点性能 | 100个test源×4噪声，G对照；预测封存后评分 | 真值对照误差、稳定输出率、区域覆盖与面积 | 新盲测、真实球型外部验证、严格conformal保证 |','',
            '实验三先复用实验二的检查逻辑，逐案例保存峰和稳定性状态，然后才评分，不重复采样。相较指定算例，批量搜索窗口扩展为覆盖整门框或由公开预测确定，避免把右上角算例窗口错误套用到全体射门。','',
            '## 3. 实验一：阻力重拟合剖面','',
            f"沿用Cd∈[0.08,0.55]的25个等距节点并加入0.22，共26层；每层基础3起点。{drag['fit_count']}个基础拟合全部成功，{drag['forward_count']}条固定参数前向对照，20/20个9起点、收紧积分容差的复核通过。",'',
            '| 组别 | 网格目标最小Cd | Y跨度/mm | Z跨度/cm | 时间跨度/ms |','|---|---:|---:|---:|---:|']
    for label,a in drag['groups'].items():
        r=a['ranges'];report.append(f"| {label} | {a['grid_minimum_cd']:.5f} | {r['y_m']['span']*1000:.3f} | {r['z_m']['span']*100:.3f} | {r['time_s']['span']*1000:.3f} |")
    report+=['',
             'F0/F1的LS数学目标相同，故共享F_LS；MAP分别运行。F1横向扫描跨度小，高度与到达时间对阻力仍有明显敏感性。网格最小层0.37375不是冻结值0.22，也没有据此回调参数。',
             '该剖面保留了有限的数据偏好，不能说数据对阻力完全没有信息；也不足以把一个网格最小层解释为已识别真实阻力。MAP目标含各自先验，不能直接跨模型排名。',
             '图选用主图1，门线变化与固定参数前向对照可作为补充图S3。原记录：[实验一报告](../整理阻力剖面实验/FINAL_DRAG_PROFILE_REPORT.md)。','',
             '## 4. 实验二：指定算例平滑峰稳定性','',
             '四个种子各64k，ESS均约4.4万，未知事件质量为0。诊断先通过，再判断峰位置。','',
             '| 带宽/m | 合并峰(Y,Z)/m | 跨种子直径/m | 32k→64k最大移动/m |','|---|---|---:|---:|']
    for b in (.05,.1,.2):
        a=stability['stability']['inside_conditioned'][str(b)];c=stability['combined_kde']['inside_conditioned'][f'64000_{b}']
        report.append(f"| {b:.2f} | ({c['y_m']:.3f}, {c['z_m']:.3f}) | {a['cross_seed_diameter_m']:.4f} | {a['max_last_doubling_shift_m']:.4f} |")
    report+=['',
             '主带宽跨种子2.24cm、加倍移动1.41cm；支持指定算例峰位置稳定。原始格网的同种子最大变化18.33cm，虽通过20cm门槛，余量较小。检查未证明无限分辨率唯一众数。',
             '图选用主图2，密度图为补充图S2；白色等密度线不能解释成90%区域。原记录：[实验二报告](../算例稳定性实验/STABILITY_REPORT.md)。','',
             '## 5. 实验三：批量主点性能','',
             '使用已有independent_200的test部分：100条独立源、四档噪声、400组观测。该数据此前已经看过，故称冻结方法补充评价，不称新盲测。每条可用输入按四种子各64k运行；完整预测封存后评分，真值不交给预测器。',
             f"380组可用，374组完成采样，6组拟合失败，20组原始输入不可用。保留{validation['raw_files']}个NPZ、{validation['raw_candidates']:,}条原始提议；拒绝候选和失败均保留。",'',
             '### 5.1 主要精度结果','',
             '| 1cm噪声、真实入框 | F1平滑峰 | G，同一批案例 |','|---|---:|---:|',
             '| 独立源数/配对观测数 | 49/49 | 49/49 |',
             f"| 中位误差/m | {main_stats['paired_peak']['median_m']:.4f} | {main_stats['paired_g']['median_m']:.4f} |",
             f"| P90/m | {main_stats['paired_peak']['p90_m']:.4f} | {main_stats['paired_g']['p90_m']:.4f} |",
             f"| RMSE/m | {main_stats['paired_peak']['rmse_m']:.4f} | {main_stats['paired_g']['rmse_m']:.4f} |",'',
             f'中位误差相对下降{100*improvement:.1f}%；按机制分层、按源整组配对bootstrap 2000次，F1−G中位误差差95%区间[{ci[0]:.4f},{ci[1]:.4f}]m。支持本批合成数据上的中位误差改善，未证明对每一机制都显著优越。',
             '49/49真实入框源通过主平滑峰检查是样本结果，不是总体100%保证；普通bootstrap在全成功时退化为[1,1]，不能据此宣称无失败风险。','',
             '### 5.2 失败和稳定输出的分母','',
             f"全部400输入中，入框条件平滑峰稳定提供{release['stable_inside_conditioned_predictions']}/400={pct(release['stable_inside_conditioned_predictions']/400)}；在380组可用输入中为{pct(release['stable_inside_conditioned_predictions']/380)}。",
             '在49条真实入框源的196组噪声观测中，193组通过（98.47%），3组采样诊断未通过。正式误差表只纳入发布主点的案例，同时必须报告这3组；未发布的原始峰不混入正式误差。',
             '四档配对193组的F1/G中位误差0.2859/0.3786m，P90为0.5605/0.6277m。配对区间和全部G案例均列在性能总表，避免只筛选F1成功案例造成误导。',
             f'原始格网在1cm层仅{raw_main}/49、四档真实入框观测中仅{raw_all}/196通过。正式结论始终是固定带宽平滑峰稳定，不能写成原始格网众数普遍稳定。','',
             '### 5.3 区域覆盖仍是限制','',
             f"1cm层：{covered_main}/77={pct(main_stats['region_coverage'])}覆盖，面积中位{main_stats['region_median_area_m2']:.4f}m²。四档：{covered_all}/308={pct(all_stats['region_coverage'])}，提供303个区域，面积中位{all_stats['region_median_area_m2']:.4f}m²；缺失区域计漏覆盖。",
             f"combined四档覆盖{pct(performance['by_mechanism']['combined']['region_coverage'])}。因此不写“达到严格90%校准”或“全部机制可靠”。",
             f"历史4k区域结果为{pct(previous['calibrated_test']['coverage'])}、面积中位{previous['calibrated_test']['median_area_m2']:.4f}m²；本次四种子合并后重新计算区域，并要求四种子的区域诊断均通过，发布口径更严格。这两张表要分开，不用新结果覆盖旧表。",
             '区域与点误差分母不同：区域评真实过线源，入框条件主点评真实入框源。置信区间包含90%也不等于证明达到90%。详表见[性能总表](性能总表.md)。','',
             '### 5.4 数值与复现证据','',
             f"批量积分只加速同一F方程。抽查29个已完成案例、{numerical['candidate_count']}条候选：相对原积分器，最大观测位置差{numerical['maxima']['observation_error_m']:.3g}m，事件位置差{numerical['maxima']['event_position_error_m']:.3g}m，时间差{numerical['maxima']['event_time_error_s']:.3g}s，log_target差{numerical['maxima']['log_target_error']:.3g}；步长减半通过。",
             '全部原始候选权重、事件编号与归一化检查通过；139项项目测试通过。测试通过是实现检查，不代替独立预测性能。',
             '原记录：[实验三完整报告](../最终主点性能实验/FINAL_POINT_EVALUATION_REPORT.md)、[结果解读](../最终主点性能实验/INTERPRETATION.md)。','',
             '## 6. 题目指定算例的答案及不确定性','',
             f"最终主点为(11,{case['primary_point']['y_m']:.3f},{case['primary_point']['z_m']:.3f})m，条件为球进入门框，主带宽0.10m。已提供前两帧坐标；没有真实后续落点，不能对该单例报告真实预测误差。",
             f"保留的直接过线条件校准椭圆面积为{case['spatial']['calibrated']['area_m2']:.4f}m²，中心({case['spatial']['calibrated']['center_yz_m'][0]:.4f},{case['spatial']['calibrated']['center_yz_m'][1]:.4f})m，阈值3.08352545。该椭圆针对直接过线，不能标成入框条件平滑峰的严格90%区域。",
             '该算例椭圆沿用保存的4k单种子协方差，主点来自四种子64k合并；两者来源和条件不同，均须注明。时间区间可放补充材料，题目主交付是位置及其不确定性。原值见[指定算例](../final_case/final_case.json)。','',
             '## 7. 论文可用结论与限制','',
             '可用表述：',
             f'> 在固定阻力和冻结先验的F1模型下，题目指定算例的0.10m固定带宽平滑峰为(2.875,1.925)m，并通过多种子与采样量稳定性检查。在观测标准差1cm的49条真实入框合成测试射门中，平滑峰中位位置误差为0.271m，相较重力抛体基线的0.374m降低约27.6%，源级配对统计支持改善。预测区域在真实过线样本上的经验覆盖为88.3%，仍存在欠覆盖，结果限于已检验的合成机制。','',
             '必须保留：固定Cd、先验和平滑尺度的条件性；49条真实入框源的样本限制；旧test已用于观察的补充评价性质；原始格网不普遍稳定；区域欠覆盖；无实测验证。',
             '不得写：毫米级真实精度；唯一连续众数；所有射门100%稳定；严格90%conformal保证；真实足球场或门将扑救成功率保证。',
             '旧开发集均值误差0.295m、旧独立过线均值误差0.318m和本次入框平滑峰0.271m不能直接相减来宣称版本提升，数据子集及估计量不同。','',
             '## 8. 论文图表与文件入口','',
             '正文建议三张主图：阻力拟合剖面、指定算例采样量稳定性、最终点误差比较；指定算例密度、区域覆盖、门线阻力变化和带宽敏感性按篇幅作为补充图。见[论文图表清单](论文图表清单.md)。',
             '正文建议两张主表：主要噪声层的配对点误差、稳定提供率与区域覆盖。完整分层表放附录或补充材料，见[性能总表](性能总表.md)。',
             '本次只完成实验汇总与论文材料准备，未重写建模报告或论文全文。原实验数据、候选和模型代码保持不变；输入指纹见evidence.json。']
    write('EXPERIMENT_SUMMARY.md','\n'.join(report))

    figdoc=['# 论文图表清单与可用图注','',
            '主图按三项实验顺序排列。所有图片直接复制自已有结果，未修改图中数据。主图3为四档噪声合计，主要1cm层数值使用性能总表，不混淆口径。','']
    for role,name,source,title,caption in figures:
        figdoc += [f'## {role}：{title}','',f'文件：[figures/{name}](figures/{name})。',
                   f'原图：[原实验图片](../{Path(source).relative_to("results").as_posix()})。','',f'建议图注：{caption}','',f'![{title}](figures/{name})','']
    figdoc+=['## 正文表格建议','',
             '- 表1使用性能总表中的1cm层F1/G配对误差，明确n=49源。',
             '- 表2使用稳定输出比例及直接过线区域覆盖；按不同分母分区或加脚注。',
             '- 四档误差、噪声分层和机制分层放附录；不能把196组噪声观测写成196条独立射门。']
    write('论文图表清单.md','\n'.join(figdoc))
    ledger='''# 结论与证据对照

| 结论 | 判定 | 证据 | 写作限制 |
|---|---|---|---|
| 本算例横向位置对Cd较稳定 | 支持 | 实验一F1 MAP扫描Y跨度1.15mm | 条件敏感性，不是总精度 |
| Cd=0.22为已识别的真实值 | 不支持 | 网格最小层为0.37375，剖面范围未被强烈排除 | 保留固定Cd的建模假设 |
| 指定算例0.10m平滑峰数值稳定 | 支持 | 实验二跨种子2.24cm、加倍1.41cm | 不等于准确或全局唯一 |
| 原始格网众数在批量数据中普遍稳定 | 不支持 | 1cm真实入框仅20/49通过 | 与平滑峰分别报告 |
| F1平滑峰相对G具有中位误差收益 | 在本批合成数据中支持 | 1cm的49源配对误差0.2709/0.3739m，区间不跨零 | 不是新盲测、不是所有机制均显著 |
| 1cm层对所有射门保证100%稳定 | 不支持 | 49/49仅为真实入框子集样本结果 | 全部400组输入仅321组提供稳定条件峰 |
| 空间区域严格达到90%覆盖 | 不支持 | 1cm 68/77，四档267/308 | 经验覆盖、承认欠覆盖 |
| 模型在实测足球或门将任务中有效 | 尚无证据 | 没有真实轨迹或门将可达性验证 | 仅陈述合成预测性能 |
| 校准使主点预测变准 | 不支持该因果解释 | 校准只作用于区域/区间后处理 | 点收益通过同数据G对照证明 |
'''
    write('结论证据对照.md',ledger)
    write('README.md','''# 三项实验汇总

请先看[实验总报告](EXPERIMENT_SUMMARY.md)。论文数字统一取[性能总表](性能总表.md)，图与图注见[论文图表清单](论文图表清单.md)，表述边界见[结论证据对照](结论证据对照.md)。

本目录只保存汇总和选图副本，完整原始数据仍在上级的三个实验目录。没有重新采样、没有模型或校准参数调整。

运行本目录的build_summary.py可从原结果重新生成；evidence.json记录输入和图源指纹。这里保留的139项测试结果是实验三的已有验证记录，本次文档整理没有新增统计检验。
''')
    evidence=dict(date='2026-11-08 Asia/Shanghai',scope='Synthesis only; no resampling, refitting or calibration.',
                  sources={s:sha(ROOT/s) for s in sources},
                  figures=[dict(role=r,file=f'figures/{n}',source=s,sha256=sha(ROOT/s)) for r,n,s,_,_ in figures],
                  primary_point_comparison=main_stats,overall=all_stats,
                  primary_paired_ci95=ci,raw_grid_primary_pass=[raw_main,49],raw_grid_pooled_pass=[raw_all,196])
    write('evidence.json',json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False))
    # Verify every local link/image in generated Markdown and every copied figure.
    missing=[];links=0
    for md in OUT.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md.read_text()):
            target=urllib.parse.unquote(target.split('#')[0])
            if target.startswith(('http:','https:')):continue
            links+=1
            if not (md.parent/target).exists():missing.append(dict(file=md.name,target=target))
    assert not missing,missing
    assert all(sha(OUT/'figures'/n)==sha(ROOT/s) for _,n,s,_,_ in figures)
    checks=dict(local_links_checked=links,broken_links=missing,figure_copies=7,figure_hashes_match=True,
                primary_point_metrics_match_per_case=True,counts_match_per_case=True,
                no_model_changes=True,no_new_experiment=True)
    write('validation.json',json.dumps(checks,ensure_ascii=False,indent=2))
    manifest={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    write('artifact_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
