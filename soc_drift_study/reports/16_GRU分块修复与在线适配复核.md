# 16. 强源 GRU 的 cell-boundary 修复复训与在线物理适配最终复核

GitHub Actions 成功运行：[38205813947](https://github.com/Ustinian1722/eVTOL-SOC/actions/runs/38205813947)。完整保留 5 折留物理电芯、3 随机种子（17/31/43）、hidden16、阶段摘要序列长度12、20个epoch及原有 SOC_proxy 源监督；**只修复原 `train_gru_proxy_crosscell.py::_make_chunks` 在串接多个源电芯后按每12个阶段打包可能混入不同电芯的序列边界**。现在每颗源电芯独立分块再堆叠，单元测试明确发现原错误并证明新分块每个样本仅属于一个物理电芯。

## 修复 GRU 自身表现（22 芯，先在芯内平均3 seeds）

| 模型 | SOC_proxy MAE 中位数（pp） | 平均（pp） |
|---|---:|---:|
| 原串接源电芯 GRU | 1.187 | 1.582 |
| **修复源电芯序列边界后的 GRU** | **1.152** | **1.542** |

安全版本相对原版本的平均配对差为 **−0.040 pp**，95% cell-bootstrap `[−0.108,+0.031] pp`，15/22 电芯改善。该差异未证明强统计显著收益，但修复了训练实现的物理分组缺陷，并让主基线更可信。训练+预测约79.5秒，完整权重、每折归一化器和逐电芯种子预测均在 Artifact。源 GRU 离线监督仍使用先前 RPT 容量归一化的 SOC_proxy，**不是全程零 RPT 的训练协议**。

## 修复后强 GRU + 常规飞行 ΔQ 标量适配

适配器运行期不读取目标 RPT 曲线、RPT 容量、SOC_proxy 标签或 SOH，仅使用前序**已完成**任务内的 `ΔQ` 和已预测的源 GRU SOC 值；根据假设容量先验修正以 SOC=100% 为支点的单一斜率参数。对同样22个芯、3 seeds、阶段端点进行成对评价：

| 方法 | 代理 MAE 电芯中位（pp） | 相对强 Frozen GRU 的平均变化（pp） | 95% 电芯 Bootstrap CI | 受益电芯 |
|---|---:|---:|---:|---:|
| **不更新（修复 GRU）** | **1.152** | 0 | [0,0] | — |
| 4次物理更新，2.7Ah 容量先验 | 1.260 | +0.133 | [−0.085,+0.402] | 9/22 |
| 4次物理价值门控，3.0Ah | 1.773 | **+0.434** | [+0.016,+0.764] | 3/22 |
| 4次物理周期更新，3.0Ah | 2.775 | **+1.255** | [+0.860,+1.628] | 3/22 |
| 每40次任务连续更新，3.0Ah | 3.690 | **+2.224** | [+1.535,+2.815] | 1/22 |
| 对照：打乱真实 ΔQ 配对 | 3.797 | +2.169 | [+1.603,+2.677] | 2/22 |
| 4次物理周期更新，3.3Ah | 4.570 | +3.129 | [+2.485,+3.693] | 1/22 |

**主结论严格保留：在强跨电芯监督预训练的 GRU 上，固定 3Ah 的无监督任务内 ΔQ 梯度/斜率更新明显有害。**此前弱 N20 Ridge 上的更新收益不能外推到强基座，这一事实不是源电芯跨边界打包错误造成的：修复后它更清楚。

## 后续研究决策

1. 以后**始终以修复后的 1.152 pp GRU 为强 Frozen 主基线**，保存该运行权重与输入特征标准化文件，并在论文中注明全部 source-cell 训练集使用全生命周期离线监督，与仅目标 N20 初始训练的信息预算不同。
2. **暂停**直接对强 GRU 应用固定 3Ah 物理相对 SOC 损失、全局仿射 Head、无约束高频在线更新和只凭漂移就触发更新。错误容量假设会把已经正确学习的老化信息覆盖掉。
3. 若仍追求强 GRU 上的 **RPT-free target adaptation**，下一研究方法应优先共享特征的**电压自监督重构与有界残差/Adapter**，而不是直接用失配容量约束强改最终 SOC 值。先证明 HOW 在修复强基座上平均、最差/后期和逐芯均无负迁移，再重新打开主动 WHEN。在线方法不得使用目标 RPT 低倍率参考。
4. 如果在当前数据上无法发现强基线上的正收益，则将论文场景严格限制为**目标新电池只允许 N20 少量标称库仑监督、没有跨电芯长期丰富源标签**。这与源 GRU 1.152 的大量离线监督不能称为公平模型网络优劣比较。
5. 以上误差仍是历史先前 RPT 容量归一化的 `SOC_proxy` 代理，并非独立真实 SOC 测量。最后必须补真正新的外部数据/电芯分组及充电协议验证，不能以在这22电芯上继续试阈值伪装为独立泛化。

## 复现

```bash
cd soc_drift_study
pip install -r requirements-sequence.txt
python -m unittest discover -s tests -p 'test_gru_boundary_safe.py' -v
python scripts/retrain_gru_boundary_safe.py --output-dir boundary_safe_artifacts
python -m unittest discover -s tests -p 'test_strong_gru_mission_adapter.py' -v
python scripts/strong_gru_mission_adapter.py --prediction-file boundary_safe_artifacts/boundary_safe_gru_predictions.csv --output-dir strong_gru_corrected_artifacts
```

注意：已提供的 `results/gru_proxy_crosscell_predictions.csv` 是旧模型历史对照，不自动覆盖；**新修复模型的权重、每芯每种子预测、CI 和所有独立审计产品均保存在本轮 Actions Artifact**。后续正式将其冻结为仓库基准时需通过审查确认并提交相应派生结果。
