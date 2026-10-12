# 11. 面向 SOC 漂移检测与在线主动更新的文献路线（RPT-Free Mission Update）

研究基线：CMU eVTOL VTC6 电池、22 个物理电芯、起飞/巡航/降落三阶段，最初 N20 任务训练；运行期**只允许普通飞行任务信号** V/I/T/阶段累计放电量的**任务内差分**进入检测和更新。严禁把 RPT 低倍率阶段 Ns 3/5/7、RPT 容量、SOH、未来信息或测试 SOC_proxy 放入更新。原始 SOC_proxy 使用历史已完成 RPT 容量标准化累计放电量，保留**离线评分**；同时设立 N20 使用固定 3 Ah 归一化的严格无 RPT 初始模型作为对照。不能将任何基于同一公式的重算称为独立 SOC 精度。

## 核心文献和相对于当前任务的启发

以下为摘要、出版社公开页面、论文官方页面所能验证的信息；没有完整开源实验条件的，不称为原论文复现。

1. **Active-learning-driven error control for data-driven state of charge estimation across the lithium battery lifecycle**, *Energy and AI* 21 (2025), 100549, DOI [10.1016/j.egyai.2025.100549](https://doi.org/10.1016/j.egyai.2025.100549)。模型分歧驱动主动学习闭环；摘要报道 4 次 retraining 与 <1.5% 平均 SOC 误差。**注意**：主动触发 ≠ 无标签更新；必须查清论文 retraining 的新 SOC 监督从何而来。仓库当前要把 When 与 How 拆开比较。
2. **Online state of charge estimation for lithium-ion batteries based on incremental learning under real-world driving conditions**, *Journal of Energy Storage* 118 (2025), 116234, DOI [10.1016/j.est.2025.116234](https://doi.org/10.1016/j.est.2025.116234)。在线顺序 ELM/增量 SVM 建立电池模型，并与自适应 UKF 结合；关键思想是用在线运行数据更新电压/动态响应模型，而不一定更新 SOC 监督头。仓库的轻量使命更新与 RLS/自适应路线目前为**原理级启发**，不等同于复现 OS-ELM/AUKF。
3. **A new online SOC estimation method using broad learning system and adaptive unscented Kalman filter algorithm**, *Energy* 309 (2024), 132920, DOI [10.1016/j.energy.2024.132920](https://doi.org/10.1016/j.energy.2024.132920)。将在线增量电池模型和自适应 UKF 结合，并对冗余更新样本进行筛选；提示应将**传感器激励充分性/新样本信息增益**作为是否更新的门。
4. **Deep domain adaptation network for transfer learning of state of charge estimation among batteries**, *Journal of Energy Storage* 61 (2023), 106812, DOI [10.1016/j.est.2023.106812](https://doi.org/10.1016/j.est.2023.106812)。显式域特征适应（含 MMD / 对抗等）且讨论有/无目标标签场景；本地对应先试 mission-only 样本统计与源域重加权，效果不好再考虑深层 MMD/GRL 网络，不能将简单 covariate weighting 称作 DDAN 复现。
5. **State-of-charge estimation across battery chemistries: A novel regression-based method and insights from unsupervised domain adaptation**, *Journal of Power Sources* 628 (2025), 235760, DOI [10.1016/j.jpowsour.2024.235760](https://doi.org/10.1016/j.jpowsour.2024.235760)。无标签回归域适应提供跨电芯/跨工况方法背景。注意全寿命目标域无标签时可迁移性的限制。
6. **An online transfer learning framework for cell SOC online estimation of battery pack in complex application conditions**, *IEEE Transactions on Transportation Electrification* (2023/2024), DOI [10.1109/TTE.2023.3324822](https://doi.org/10.1109/TTE.2023.3324822)。明确指出目标空间变化可能导致负迁移，提出时变部分目标空间和在线转换机制。对应当前全寿命模型变化，需做 SOC 范围条件化匹配、负迁移检查。
7. **Continual learning for online state of charge estimation across diverse lithium-ion batteries**, *Journal of Energy Storage* 117 (2025), 116086, DOI [10.1016/j.est.2025.116086](https://doi.org/10.1016/j.est.2025.116086)。TCN + PNN 的持续学习减少跨电池任务遗忘；但是否需要目标 SOC 监督应按论文实验具体协议判定，不可声称它自然解决完全无标签更新。
8. **State-of-charge estimation of lithium-ion battery: Joint long short-term memory network and adaptive extended Kalman filter online estimation algorithm**, *Journal of Power Sources* 604 (2024), 234451, DOI [10.1016/j.jpowsour.2024.234451](https://doi.org/10.1016/j.jpowsour.2024.234451)。神经网络 SOC 估计 + adaptive EKF 状态融合，提示可把慢变模型的在线识别与快速 SOC 估计模块拆开。
9. **Online Parameter Identification and State of Charge Estimation of Lithium-Ion Batteries Based on Forgetting Factor Recursive Least Squares and Nonlinear Kalman Filter**, *Energies* (2018), [Publisher](https://www.mdpi.com/1996-1073/11/1/3)。RLS 在线辨识 ECM 参数 + 滤波 SOC；这是未来**真正电压残差驱动在线辨识**的重要物理基线。
10. **Lithium battery state of charge estimation based on improved variable forgetting factor recursive least squares method and adaptive Kalman filter joint algorithm**, *Journal of Energy Storage* 100 (2024), 113392, DOI [10.1016/j.est.2024.113392](https://doi.org/10.1016/j.est.2024.113392)。自适应忘记因子的在线参数更新，而非直接用 RPT 做输出层仿射偏置。
11. **Continual Test-Time Domain Adaptation**, *CVPR* (2022), [Open access](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)。CoTTA 教师平均、增强平均与源权重随机恢复防止无标签自训练的误差累积；目前本地 pseudo 自训练只是**简化且不等价的线性代理**。要复现需改成可在线优化的神经序列 backbone。
12. **In Search of Lost Online Test-Time Adaptation: A Survey**, *IJCV* 133 (2025), [Publisher](https://link.springer.com/article/10.1007/s11263-024-02213-5)。归纳 normalization / pseudo-label / 数据增强 / 结构更新路线；TTA 的**类别标签熵最小化**不能直接搬到连续 SOC 回归。应设计电量守恒、电压重构或动力学残差类任务。
13. **Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection**, *ICCV* (2025), [Official PDF](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Park_Hybrid-TTA_Continual_Test-time_Adaptation_via_Dynamic_Domain_Shift_Detection_ICCV_2025_paper.pdf)。根据动态漂移选择适配幅度，启发门控/源域保护，但本地测试不可把概念等同于原 Hybrid-TTA。
14. **A multi-task transformer approach for lithium-ion battery pack health state estimation using self-supervised reconstruction and online fine-tuning**, *Journal of Energy Storage*, [Publisher](https://www.sciencedirect.com/science/article/pii/S2352152X25036357)。SOH 估计中辅助重构 + unlabeled test-time training；目前仓库具有电压原始测量，下一阶段可构建**飞行段电压重构预训练 + 无标签在线损失更新**，这是高优先级深度网络方案。
15. **Evaluation of Test-Time Adaptation Under Computational Time Constraints**, *ICML* (2024), [PMLR](https://proceedings.mlr.press/v235/alfarra24a.html)。比较方法时须计入真实在线推理时间与更新成本；不仅报代理误差，也需报更新数/时长/参数。

## 研究分类（与“更新是否需要新 SOC 标签”分开）

| 类别 | 检测/适配机制 | 运行期新 SOC 标签要求 | 本地实验实现级别 |
|---|---|---|---|
| Frozen / Periodic / Random | 不更新 / 固定节拍 / 随机触发 | 无 | 完成 |
| 电压残差漂移监测 | Vmin 预测残差阈值、CUSUM、Page-Hinkley 风格 | 无 | 简化线性已实现 |
| 物理一致性失配 | 同任务 SOC 相对差与 ΔQ/3Ah 约束失配 | 无 | 简化阶段端点已实现 |
| Test-time normalization | 仅校正输入统计 | 无 | 简化固定补偿已实现 |
| 无监督 covariate shift | 只用目标未标记分布重新加权 N20 监督数据 | 无 | 简化加权 Ridge 已实现 |
| Anchored pseudo self-training | 以初始预测 + 任务内 ΔQ 推导伪标签 | 无，但需冻结教师/源约束 | 简化线性已实现 |
| Physics-only constrained adaptation | 约束网络在同次任务内的 SOC 差 | 无 | 任务内斜率 + Ridge 约束已实现 |
| Source-replay guard | 过去飞行物理残差作为门控+初期监督误差保护 | 无 | 简化门控已实现 |
| Incremental ELM/SVM + AUKF | 学习在线动态电压模型 + 状态滤波 | 可无新 SOC 标签，但依赖具体辨识/可观测假设 | 尚未完整实现 |
| ECM-RLS + UKF/EKF | 在线电压模型辨识 + SOC 滤波 | 不必有新 SOC 真值，但需 OCV/初始状态先验与足够激励 | 尚未完整实现 |
| CoTTA / PNN / MMD TTT | 可训练神经表征和辅助无监督目标 | CoTTA/MMD 可无，PNN 必须审视其监督假设 | 尚未完整实现 |

## 实验协议及论文证据的硬边界

- **任务内差分** ΔQ 可从工作电流积分，但 SOC_proxy 的**绝对容量分母**来自先前 RPT。物理一致性方程只能约束 SOC 的相对变化，**不能自动辨识绝对 SOC 偏置或当前容量**。
- 预先固定所有阈值、强度和预算，严格 predict → observe completed mission → detect → optionally adapt → next mission 生效。
- 此轮最多 B=4 次尝试（stream 例外），无标签触发不保证用满四次，需报告实际次数；尚不是预算严格匹配的 When-to-Update 结论。
- 两条 N20 初始化轨道：原本 SOC_proxy 标准化的训练，和标称 3Ah 的任务电量初始训练；若用户要求全程不碰 RPT，**只能采用第二轨作为模型本身**，前者留作历史协议对照；两轨的 RPT 仅用于离线评价 SOC_proxy。
- MAE、RMSE、晚期 MAE、P95 循环 MAE、逐电芯和 cell-bootstrap 统计，不允许按随机窗口重采样进行显著性宣称。
- 这是研究路线 **广覆盖探索**，不是声称重现上述 15 篇不同模型，也不是根据留出测试成绩选择新发明方法的独立外测。若要定稿，需另设模型选择/验证电芯与全新最终测试。
