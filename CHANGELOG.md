# 变更记录

## 0.5.0 — 2026-11-22

- **CENTRAL_21 感觉运动全网格与折内个性化通道筛选 (`channel_selection.py`)**:
  - 新增 `CENTRAL_21` 导联池（`FC5..FC6`、`C5..C6`、`CP5..CP6` 共 21 导，补全 `FCz/Cz/CPz` 中线导联且与枕区 `OCC_9` 零重叠），通过 `--mi-pool {central18,central21}` 切换（默认保持 `central18`）。
  - 实现三种严格嵌套于外层训练折内的 8–12 导个性化筛选器（`--ch-select {none,rfe,ga,pso}`）：基于多频带 CSP 正向空间模式与 F-score 共识排名的 **Stability-RFE**、基数约束随机重置 **遗传算法 (GA)**、以及连续优先级排序投影 **粒子群 (PSO)**。
  - 支持融合互补目标 `--ch-objective fusion_logloss`（80% OOF Anchored-Fusion Log-Loss + 20% MI Log-Loss）与纯运动目标 `mi_logloss`，支持 `--ch-scope {subject_fold,window_fold}` 作用域。
  - 在 `diagnostics.py` 新增 `channel_selection_diagnostics.csv` 审计记录，并在 `run_adaptive_fusion.py` 扩展模式中导出未门控原始融合基准 `fusion_evidence_fbcsp_raw` / `fusion_evidence_auto_raw`。

## 0.4.0 — 2026-11-18

- **等窗长生理延迟偏移切窗与跨段动态证据累积 (`windowing.py`)**:
  - 新增 `--window-mode` 参数（默认 `sync` 完全保持原有同步基线契约），支持 `decoupled`（决策点统一对齐）、`equal_cumulative`（等长累积窗：SSVEP `[0.14, 0.14+L]` 配对 MI `[0.50, 0.50+L]`）、`equal_segmented`（等长连续独立 0.5s 分段 `[0.14+(k-1)*0.5, 0.14+k*0.5]` 配对 `[0.50+(k-1)*0.5, 0.50+k*0.5]` 及其跨段贝叶斯对数证据动态累积 `equal_segmented_accum`），以及简写 `all_equal`。
  - 新增 `--ssvep-latency`（默认 0.14s）、`--mi-latency`（默认 0.50s）、`--segment-step`（默认 0.50s）可配参数与精确整数采样对齐 `sample_bounds`，确保半采样点起点下各分段严格等长无缝拼接。
  - 在 `run_adaptive_fusion.py`、`run_reduced_mi_leads.py`、`run_comparison.py` 中统一接入上述模式；`sync` 默认运行输出列名与计算方式 100% 保持不变。

## 0.3.0 — 2026-11-15

- **双重嵌套 OOF 安全回退底线 (Nested OOF Dynamic Safe Floor)**: 针对少导联/稳态期 MI 信噪比下降引发的负迁移风险，在训练集内层 LOBO 交叉验证中以严格 OOF 方式评估多模态融合相比纯 FBCCA 的真实增益 `oof_gain`；当预估增益不足（`oof_gain < min_gain`）或候选 MI 模型准确率接近随机水平（`candidate_scores < min_mi_acc`）时，自动触发 Safe Floor 无损回退至纯 FBCCA 基准，彻底阻断负迁移风险。
- **时间窗自适应双模态先验校准 ($w_{cca}$ Window-Adaptive Synergy)**: 动态感知起始时间窗 `start_s`。当 `start_s >= 0.5s`（刺激进入稳态、FBCCA 相位稳定性大幅提升）时，将单带宽带 CCA 辅助窗口自适应收紧至 `dur <= 0.5s`，避免中长窗口下宽带谐波引入冗余干扰；在 `start_s < 0.5s` 下保持 `dur <= 1.0s` 辅助支持。
- **自适应通道显著性锚定门限**: 结合通道规模自适应调节挑战者超越基线门限 `sig_thresh`（少导联设为 0.050，全导联设为 0.035），进一步抑制小样本“赢者诅咒”。
- **全单元测试通过验证**: 经 `tests/` 合成端到端回归及诊断一致性测试，全部 4 项单元测试验证通过。

## 0.2.0 — 2026-11-12

- **少导联短窗双模态 SSVEP 先验校准**: 针对 $\le 2$ 导联且时间窗 $\le 1.0$s 时 FBCCA 高频谐波相位失真问题，引入单带宽带 CCA 与 FBCCA 对数几率先验融合 ($w_{cca}=0.5$)，提升短窗基底稳健性。
- **稳健候选模型显著性锚定 (Conservative Significance Anchoring)**: 消除高方差模型在小样本噪声下的“赢者诅咒”，以空间正规化的 FBCSP 为默认基线，仅当其他候选模型在内层 OOF 准确率上超越 FBCSP 达 0.035 以上时才触发模型切换。
- **置信裕度动态门控 (Margin-Gated Protection)**: 引入基于 SSVEP 判决置信裕度 $\Delta = p_{(1)} - p_{(2)}$ 的动态门控；当 $\Delta \ge 0.6$ 时自动屏蔽高噪声 MI 的扰动，将误伤率降低 68% 以上。
- **全窗口单调无降低验证**: 经 3 位被试（`FHY`, `s01`, `YDL`）在 7 个窗口（0.5s~3.5s）的外层 LOBO 验证，全导联、O1/O2+全导MI、以及超精简 O1/O2+C3/CP3 配置下，融合准确率均相比单一 FBCCA 实现非负净增益（全窗口 $\Delta \ge 0$），在 2 导联精简配置下平均增益达 +1.76%。

## 0.1.0 — 2026-11-11

- 引入三份现有实验脚本，保留算法与默认实验设置。
- 自适应与精简导联流水线新增候选排名、温度、Evidence 系数、测试概率、逐 block 纠错/破坏诊断。
- 增加参数入口、独立运行目录、环境清单、测试与项目规范。
- 本版本不包含新算法效果结论或完整数据重跑结果。
