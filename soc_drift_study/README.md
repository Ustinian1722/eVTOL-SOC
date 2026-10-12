# eVTOL SOC Drift Study

这是对 CMU eVTOL Battery Dataset 的独立实验室研究目录。根目录 `VAHxx.csv` 和阻抗 CSV 是只读源数据；代码、模型、日志、派生结果和图均在本目录。数据为 VTC6 单体 eVTOL 任务试验，不是农业无人机现场飞行数据。

**解释边界：**原始数据没有独立 BMS 或电化学 SOC 真值。此项目的所有 SOC MAE/RMSE 均相对先验 RPT 容量与放电积分构造的 `SOC_proxy`，不能称为真实 SOC 精度。低倍率 RPT 标签也不等于高功率任务可用容量。当前维护回放使用 Ridge；GRU 只完成跨电芯代理估计实验。

## 报告

- [01 数据审计](reports/01_数据审计.md) · [02 SOC 标签、Ridge 与 GRU 基线](reports/02_SOC标签与基线.md) · [03 生命周期漂移](reports/03_生命周期漂移分析.md)
- [04 漂移检测](reports/04_漂移检测实验.md) · [05 主动维护](reports/05_主动维护实验.md) · [06 更新与成本](reports/06_模型更新与成本分析.md)
- [07 完整结果](reports/07_完整实验结果.md) · [08 硕士论文框架](reports/08_硕士论文研究框架.md) · [09 最终决策](reports/09_最终研究决策.md)
- [10 RPT 更新负迁移及支持域审计](reports/10_RPT更新迁移诊断.md)：独立 GitHub Actions 复核 22 电芯 B=4 更新、迁移失效、收缩与门控，所有误差仍是 SOC_proxy。

## 环境与复现

基础依赖：`python -m pip install -r requirements.txt`。跨电芯 GRU 需要 TensorFlow CPU 2.20：`python -m pip install -r requirements-sequence.txt`。本次环境与版本见 [`logs/environment.json`](logs/environment.json)。

从本目录运行；全量审计会顺序读取约 1.4 GB 主表，不下载或改写源数据。

```powershell
python -B scripts/audit_dataset.py
python -B scripts/make_audit_products.py
python -B scripts/extract_anomaly_rows.py
python -B scripts/build_proxy_baselines.py
python -B scripts/train_gru_proxy_crosscell.py
python -B scripts/evaluate_source_ridge_full_crosscell.py
python -B scripts/analyze_gru_proxy_results.py
python -B scripts/proxy_drift_maintenance.py
python -B scripts/offline_oracle_upper_bound.py
python -B scripts/rpt_head_replay_sensitivity.py
python -B scripts/analyze_maintenance_results.py
python -B scripts/validate_study.py
```

维护预算、阈值、随机种子、校正边界和 Bootstrap 参数见 [`configs/proxy_maintenance.json`](configs/proxy_maintenance.json)；基线配置见 [`configs/proxy_baseline.json`](configs/proxy_baseline.json)。逐步命令、数据计数与解释限制记录在 [`logs/experiment_log.md`](logs/experiment_log.md)。测试集阈值不做搜索；`offline_oracle_upper_bound.py` 使用未来代理误差，仅为不可部署的事后上界。

## 产物位置

- `scripts/`：审计、建模、因果回放、汇总和校验脚本。
- `results/`：逐循环/逐电芯 CSV、机器可读标签与实验元数据。
- `models/`：Ridge 与留组 GRU 权重及训练标准化参数。
- `figures/`：容量/阻抗、误差、触发时间线和维护对照图。
- `logs/`：环境、GRU epoch 损失和实验运行记录。

## 数据来源

引用 CMU [eVTOL Battery Dataset](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830) 和 Bills 等人的 [Scientific Data 论文](https://www.nature.com/articles/s41597-023-02180-5)（2023）。历史 FiberResearch 分支的结果与标签/传感器协议不同，且 SRA 依赖此数据集中没有的 FBG；这里只链接用于方法背景，不宣称完整复现。
