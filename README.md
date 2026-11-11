# MI–SSVEP Algorithm

四分类单指 MI / SSVEP 离线分析：基础融合基准、被试先验与折内自适应融合、精简 MI 导联。当前版本增加可追溯诊断，**没有改变 Auto 按内层 MI 准确率选模的规则，也不宣称提升准确率**。

## 安装与运行

建议使用独立 Python 环境。仓库不包含 EEG、预处理缓存或个人试次结果。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_adaptive_fusion.py --cache-dir "E:/BUAA/ssvep_mi/analysis_data/three_subjects_20261027_central_raw/cache"
.venv/Scripts/python run_reduced_mi_leads.py --cache-dir "E:/BUAA/ssvep_mi/analysis_data/three_subjects_20261027_central_raw/cache" --n-jobs 6
.venv/Scripts/python run_comparison.py --cache-dir "E:/BUAA/ssvep_mi/analysis_data/three_subjects_20261027_central_raw/cache"
```

先测试一个窗口：追加 `--subjects YDL --starts 0.5 --durations 0.5`。可指定 `--out-dir`，但必须是尚不存在的目录；默认生成独立时间戳目录，避免覆盖历史结果。

缓存文件名为 `YDL_preprocessed.npz`、`YZL_preprocessed.npz`、`FHY_preprocessed.npz`。必需字段：`ds_epochs`（试次×时间×通道，250 Hz）、`y`（0–3，thumb/index/middle/little）、`blocks`、`trial_ids`、`ons`（降采样后的起点样本索引）、`ch_labels`。每个训练划分必须覆盖四类；缓存须包含足够窗口长度与所需通道。只加载可信缓存，原实现使用 `allow_pickle=True`。预处理生成器不在本仓库范围内。

## 项目入口

| 脚本 | 功能 |
|---|---|
| `run_comparison.py` | 18 导 MI，2/3 导 SSVEP，多种融合基准 |
| `run_adaptive_fusion.py` | 18 导 MI，O1/O2，固定 FBCSP / Prior-Best / Auto |
| `run_reduced_mi_leads.py` | 7 个 2–4 导方案 + 18 导基准，按被试与窗口并行 |
| `diagnostics.py` | 纯记录逻辑、纠错/破坏统计与 CSV 输出 |
| `runtime.py` | 参数、窗口配置及环境清单 |

## 新增输出

- `fold_diagnostics.csv`：每外层折×候选模型一行；MI 内层准确率、稳定排序、第一/第二名、差距、并列名单、最终选择、MI/SSVEP 温度、对应候选 Evidence 的 a/b/四类偏置。
- `trial_predictions.csv`：每试次×方法一行；真实标签、预测、四类概率、FBCCA 概率及纠错/破坏标记。包含四个 MI、四个候选 Evidence 和 Auto；自适应脚本另保存 Prior-Best 与固定 Evidence 别名。原有 Stacking/Fixed 仍在原汇总表中。
- `block_metrics.csv`：每外层测试 block×方法一行；准确率、纠错数、破坏数、净增益计数和准确率差（0–1 单位，乘 100 为百分点）。
- `run_manifest.json`：运行参数、Python 与依赖版本。

精简导联的三张诊断表位于 `diagnostics/{subject}_start{start}_dur{duration}/`；包含 `mi_channels` 区分导联方案，避免并行写同一文件。原有汇总文件名、列及计算方式保留。

逐试次关联键为 subject / window_start_s / duration_s / mi_channels / test_block / trial_id / method；fold 表以 candidate 替代 method。

## 评估边界

外层留一 block 测试，内层 MI 折外预测选模。新增候选融合只用外层训练标签拟合，外层真实标签只用于诊断统计，不反馈选模。平分保留原候选插入顺序。温度校准与融合复用外层训练信息，不能称为完全独立的逐层交叉拟合。

Prior-Best 映射的独立历史来源尚未核实；若由当前完整数据获得，应作为探索性参考。中央区原始信号没有排除 SSVEP 成分，不能把全部收益归因于纯 MI。

诊断是离线结果；不按外层测试结果挑选模型。`net_gain_count = corrections - damages = method_correct - fbcca_correct`。窗口相互重叠，不能当成独立样本做显著性判断。

## 验证与协作

```powershell
.venv/Scripts/python -m unittest discover -s tests -v
```

测试含合成 EEG 端到端运行、诊断开关预测一致性、并列选模、试次关联及纠错恒等式；不依赖参与者数据。参见 [验证记录](docs/validation.md)、[实施计划](docs/implementation-plan.md)、[后续路线](docs/roadmap.md) 和 [贡献规范](CONTRIBUTING.md)。
