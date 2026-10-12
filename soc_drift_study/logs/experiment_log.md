# 实验日志

日期：2026-10-12。所有脚本从 `soc_drift_study/` 执行；原始 `VAHxx.csv` 与阻抗 CSV 只读，派生文件均写在本目录。

## 数据和标签

```powershell
python -B scripts/audit_dataset.py
python -B scripts/make_audit_products.py
python -B scripts/extract_anomaly_rows.py
python -B scripts/build_proxy_baselines.py
```

全量扫描 22 个主表、15,309,317 行和 21 个阻抗表；产生结构、异常、候选 RPT、容量/阻抗图和 `SOC_proxy`。保留 62,563 个阶段标签、20,855 个任务循环；容量锚点严格早于每个标注循环。检查项目见 `results/audit_metadata.json`、`results/label_audit.json` 和 `results/anomaly_summary.json`。

## 估计模型

```powershell
python -B scripts/train_gru_proxy_crosscell.py
python -B scripts/evaluate_source_ridge_full_crosscell.py
python -B scripts/analyze_gru_proxy_results.py
```

阶段摘要 GRU 使用 5 折电芯留组、3 个种子、20 epoch、16 隐藏单元和 12 步序列，训练用源电芯的全部已标注生命周期。留出电芯未参与标准化或权重训练；测试只评分目标 N20 之后，另行报告目标 N20 仿射校正。训练用时约 69.6 秒（CPU），写出 15 组权重/标准化器、epoch 日志、逐阶段预测和配对比较。另以相同折训练完整源生命周期 Ridge 控制。

## 漂移和维护

```powershell
python -B scripts/proxy_drift_maintenance.py
python -B scripts/offline_oracle_upper_bound.py
python -B scripts/rpt_head_replay_sensitivity.py
python -B scripts/analyze_maintenance_results.py
```

主回放共 814 组电芯/策略评估（22 电芯、冻结、B=2/4/6、5 随机种子和预注册检测参数）。模型分歧、电压残差、CUSUM、Page-Hinkley 风格、ADWIN-like 与混合策略均使用相同代理 Ridge、预算和候选参考机会。Oracle 明确使用测试期代理标签，只作事后上界。历史回放敏感性另有 132 组评估。配置和执行参数在 `configs/proxy_maintenance.json`、`results/proxy_maintenance_metadata.json`、`results/offline_oracle_metadata.json`。

## 主要解释边界

所有 MAE/RMSE 都针对 `SOC_proxy`，不是独立真实 SOC。电芯是统计重采样单位，Bootstrap 10,000 次、种子 20261012。没有测试期阈值搜索；Oracle 使用未来误差且不可在线实现。现有 maintenance replay 使用 N20 Ridge，不应外推成 GRU 更新结果。完整检查由 `python -B scripts/validate_study.py` 执行，结果写到 `results/pipeline_validation.json`。

最终检查：验证脚本 23/23 项通过；README 和 9 份报告的本地 Markdown 链接检查发现 0 个失效链接。
