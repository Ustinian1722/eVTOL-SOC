# eVTOL SOC Drift Study

这是对 CMU eVTOL Battery Dataset 的独立实验室研究目录。根目录 `VAHxx.csv` 和阻抗 CSV 是只读源数据；代码、模型、日志、派生结果和图均在本目录。数据为 VTC6 单体 eVTOL 任务试验，不是农业无人机现场飞行数据。

**解释边界：**原始数据没有独立 BMS 或电化学 SOC 真值。此项目的所有 SOC MAE/RMSE 均相对先验 RPT 容量与放电积分构造的 `SOC_proxy`，不能称为真实 SOC 精度。低倍率 RPT 标签也不等于高功率任务可用容量。当前维护回放使用 Ridge；GRU 只完成跨电芯代理估计实验。

## 报告

- [01 数据审计](reports/01_数据审计.md) · [02 SOC 标签、Ridge 与 GRU 基线](reports/02_SOC标签与基线.md) · [03 生命周期漂移](reports/03_生命周期漂移分析.md)
- [04 漂移检测](reports/04_漂移检测实验.md) · [05 主动维护](reports/05_主动维护实验.md) · [06 更新与成本](reports/06_模型更新与成本分析.md)
- [07 完整结果](reports/07_完整实验结果.md) · [08 硕士论文框架](reports/08_硕士论文研究框架.md) · [09 最终决策](reports/09_最终研究决策.md)
- [11 非 RPT 在线更新相关论文与方法路线](reports/11_非RPT工况主动更新文献与路线.md) · [12 飞行数据电压自监督更新](reports/12_飞行工况在线电压更新实验.md)
- [13 非 RPT 在线 SOC 更新 13 路实验](reports/13_非RPT_SOC主动更新13路结果.md) · [14 更新价值与强 GRU 反证](reports/14_无RPT更新价值门控诊断.md) · [15 跨电芯学习更新价值 NO-GO](reports/15_学得更新收益触发与跨电芯失效.md)
- [16 强 GRU 源电芯分块修复与反证](reports/16_GRU分块修复与在线适配复核.md) · [17 强 GRU 飞行电压自监督在线校正](reports/17_强GRU飞行电压自监督在线校正.md)
- **[18 同主题主动监督校准文献与WHEN成本研究方案](reports/18_主动监督校准文献与维护成本路线.md) · [19 首轮三点SOC监督校准的WHEN×COST对照](reports/19_三点SOC监督校准_WHEN成本首轮结果.md)**
- **[20 全任务SOC代理曲线可用性、14/8跨芯校准ROI决策](reports/20_完整任务曲线审计与跨芯ROI维护决策.md)**

**当前硕士开题主线（2026-10-12）正式转向 WHEN × COST：**日常正常飞行不增加 SOC 监督；漂移/收益触发一次**假设的同倍率SOC标定数据采购**后，允许以其标签进行固定有监督校准；重点比较哪些时机维护最划算，而非不断研发无标签HOW。报告19已确认：在22电芯的3阶段末端标签**先导**实验上，预定周期 MAE 中位2.270pp < 冻结2.523pp，B≤4事后Oracle2.098pp，Oracle比周期平均还低0.159pp；现有主动触发**未胜过周期**。成本权重仅是假设性MAE等价成本，不是现场人民币数据。仓库暂没有整次任务高频独立真实SOC曲线，必须扩展完整曲线级有监督校准后再用于毕业论文定稿。报告14–17属于之前“无标签HOW探索”的历史负/正结果，不能与新研究协议混为一谈。

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
