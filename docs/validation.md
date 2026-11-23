# 验证记录 — 2026-11-23

- Windows / Python 3.14.7；NumPy 2.5.4、SciPy 1.18.1、pandas 3.1.1、scikit-learn 1.9.1、pyriemann 0.11、joblib 1.6.0。
- 全量单元测试与合成数据端到端测试（`.venv\Scripts\python.exe -m unittest discover -s tests -v`）：共运行 21 项测试全部通过（88.82s），无跳过或放宽断言。
- 本次新增回归测试（`tests/test_signal_fixes.py`）覆盖：
  - 缓存健壮性与窗口边界强校验（拒绝非法标签 `{0..3}` 外数值、`NaN`、缺失通道与越界时间窗）。
  - 整段零相位滤波 `epoch_zerophase` 对短窗（0.10s / 25 点）无边界报错，以及单向因果 `causal` 模式对未来样本零泄露。
  - 宽带 CCA（`(6, 45) Hz`）抗慢漂移验证、`project_out_ssvep` 正交剔除 10/20 Hz 谐波验证、以及信号域 `apply_subset_car` 与协方差域 `apply_cov_car` 等价恒等式验证。
  - `FBPowerLDAModel` 与 `TangentLogRegModel` 子集内 `c / trace(c)` 缩放不变性、`RCSPModel` 4 导联 8 滤波器收敛性、及缺类 `align_proba_4class` 防护。
  - `subject_fold` 通道选择结果与 `--durations` 传入组合及 `--starts` 顺序严格无关。
  - `run_comparison.py` 中 `FeatureFusionModel` 折内 48 维固定槽位 OOF 特征提取与跨段特征平均对齐。
- 本次未在真实参与者缓存上重跑全部窗口与全部被试；真实数据效果需在独立环境中复跑验证。

# 历史验证记录 — 2026-11-11

- Windows / Python 3.14.7；NumPy 2.5.4、SciPy 1.18.1、pandas 3.1.1、scikit-learn 1.9.1、pyriemann 0.11、joblib 1.6.0。
- 单元测试覆盖并列、错位输入、纠错/破坏恒等式、输出目录不可覆盖和重复被试去重。
- 合成数据：24 试次、3 blocks、4 类；精简流水线全部 8 导联方案诊断开关的汇总结果相同。自适应逐试次 Evidence 准确率与汇总一致。
- 真实缓存抽样：YDL / 起点 0.5 秒 / 窗长 0.5 秒，新旧自适应脚本在相同环境中运行，全部 15 方法汇总 CSV 内容一致。旧脚本仅替换入口参数与输出路径。
- 真实抽样输出：32 条候选诊断、1,920 条逐试次记录、96 条 block 指标。数据未上传。
- 独立代码审查确认新增诊断未反馈外层标签；已修复重复被试和浮点窗长格式造成的并行目录碰撞。
- 本次未完成全部被试×窗口真实数据重跑，未验证实时使用，未提出算法增益结论。依赖版本更新可能影响与历史 CSV 的复现。
- 本机依赖导入提示 matplotlib 字体缓存目录不可写；计算与测试仍完成，未修改用户目录权限。
