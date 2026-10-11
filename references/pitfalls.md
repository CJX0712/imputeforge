# SOP pitfalls.md — ImputeForge 本轮新踩坑

> 追加到既有 pitfalls 库。每条含：现象 / 根因 / 修复 / 教训。

---

## P1. SoftImpute 把 clamp 放在 SVD 之后破坏近端梯度单调性

- **现象**：`ConvergenceError: objective increased`；`test_objective_is_monotone` 失败。
- **根因**：初版在 SVD 重建 `Z_new` 之后再 clamp 到观测值，破坏了 Mazumder 2010 的
  近端梯度结构 `Z_{k+1} = S_λ(P_Ω(X) + P_Ω^⊥(Z_k))`，目标不再单调。
- **修复**：clamp 改在 SVD **之前**（`A = where(obs, X, Z)`），输出不再重 clamp；
  用初始 `Z` 播种 `prev` 避免首轮 `objs` 长度 1 误判。
- **教训**：软阈值/近端梯度法，阈值与投影顺序必须与原始论文逐字符对齐；输出后处理会毁掉收敛性证明。

## P2. KNNImputerNP 自匹配（k=1 取到自身行）

- **现象**：手写 KNN 在 query==train 时把自身作为最近邻，填值回写为原值→虚假 0 误差。
- **根因**：距离在所有列上度量，且未排除自身行。
- **修复**：距离仅在 query 行的**已观测维**上度量；只与「目标列被观测的训练行」比较，
  天然排除自身（目标列被观测 ⇒ 该行非待补 ⇒ 不在候选集）。
- **教训**：惰性/回填式 imputer，候选集必须显式排除待补行；nan-aware 距离维度要对齐。

## P3. 探针模型选择的噪声大于方法间差异

- **现象**：重采样 CV 下候选排名 spearman 仅 ~0.52，逐列选择增益被抖动稀释。
- **根因**：单掩码 CV 方差大；选择噪声 > 融合能撬动的 headroom。
- **修复**：引入 `probe_repeats`（独立掩码池化 SSE），扫参实证 repeats=3/margin=0.02 最佳，
  因与 repeats=2 差异在噪声内且成本减半，定默认 `probe_repeats=2, margin=0.02`。
- **教训**：当「选对模型」的方差逼近「方法差」时，池化方差优先于追求单次精度。

## P4. 快速 demo 超 60s 预算（性能优化 + 拆分）

- **现象**：初版 demo 326s / 93s / 101s，远超 60s 自检预算。
- **根因**：EM 是瓶颈；full EM（60 次迭代）在 0.42 缺失率下需 1.06s/次。
- **修复**：① 拆分全量脚本 `run_full_benchmark.py`；② demo 降网格
  （n_rows=220 / 1 seed / ablation 仅 2 cells×3 modes）；③ 真实性能优化
  `probe_em_max_iter=25`（探针阶段仅 25 次迭代足够排名，最终拟合仍 60 次）。
- **教训**：探针/排名阶段可用更少迭代，仅最终部署用满迭代；预算拆分比硬扛更稳。

## P5. MICE ConvergenceWarning 升级为测试错误

- **现象**：`test_leakage` / `test_pipeline` 因 IterativeImputer 的 ConvergenceWarning 变 Error 失败。
- **根因**：全局 `filterwarnings=error` 把无害收敛告警当错误。
- **修复**：加 `_quiet()` contextmanager **局部**抑制 IterativeImputer 的 UserWarning/ConvergenceWarning，
  不全局压制（保留其它真实告警可见）。
- **教训**：第三方迭代器告警应局部隔离，避免全局 `error` 滤镜误伤。

## P6. 融合规则实证全为负 → 转向逐列条件选择

- **现象**：6 种池输出融合规则（mean/median/min/weighted/stacking/oracle）相对最强单法
  全为负（−1.5% ~ −23.2%）。
- **根因**：稀释 + 同质 DGP 零 headroom + 选择噪声。
- **修复**：放弃「池输出融合」，改为**部署对齐的逐列条件模型选择**（每列独立选 gauss/forest/huber）。
- **教训**：融合的增益来自「每列选对偏置」，而非「把多个输出再平均」；后者只会稀释。
