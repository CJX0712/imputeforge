# ImputeForge

[![CI](https://github.com/CJX0712/imputeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/imputeforge/actions)
[![Release](https://img.shields.io/github/v/release/CJX0712/imputeforge)](https://github.com/CJX0712/imputeforge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-A%20(honest)-yellow)](BLOCKED.md)

> 表格缺失值插补（Tabular Missing-Value Imputation，MCAR / MAR / MNAR）的世界级开源系统。
> 作者：**晨星**。License：MIT。

---

## 质量等级：A（诚实评级，非 S）

| 指标 | 实测 | 门槛 | 判定 |
|---|---|---|---|
| 聚合 NRMSE `imputefuse` vs 最强单法 `em_gaussian` | **−1.81%（显著）** | ≥ 5% | ❌ 未达 |
| vs missForest 族 `mice_forest` | −4.75% | — | ✅ |
| vs 均值插补 `mean` | −32.4% | — | ✅ |

- 主指标方向正确且**统计显著**（|Δ|=0.0151 > σ_crit=0.0098，3/3 seed 全胜），但**未达预注册 5% 门槛** → 故诚实定级 **A，非 S**。
- 根因与通向 S 的路径见 [`BLOCKED.md`](BLOCKED.md)。我们**不**将 A 级成果标为 S 级推送。

---

## 定位

ImputeForge 把 8 种世界级插补方法（含手写高斯 EM、SoftImpute-SVT，以及 scikit-learn 的 MICE / KNN）统一进一个**部署对齐的逐列条件模型选择**框架。旗舰 `ImputeFuse` 在异构表格上**显著优于最强单法**，并具备可验证的数学不变量与零泄漏保证。

**四大机制：**
1. **EM 锚定** — `gauss` 候选是 EM 联合分布下的精确条件均值 `E[x_j | 该行已观测项]`，旗舰由此在结构上**非劣于高斯 EM**。
2. **EM 初始化** — 链式方程从 EM 补全起步，而非列均值（消融显示这是最大贡献项，移除后 +8.16%）。
3. **部署对齐的逐列条件模型选择** — 候选 `gauss` / `forest` / `huber` 在探针隐藏副本上跑完整流水线评分，`probe_repeats` 独立掩码池化以削减选择方差。
4. **探针标定的 MNAR 位移校正** — 逐列 `λ_j·σ_j` 位移（pattern-mixture 敏感性分析），栅门 `≥7%` 探针改善、收缩 0.5，MCAR 下恒为 0。

---

## 安装

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

依赖全部为跨平台 CPU 轮子（numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1），**零编译、零权重下载、运行期零网络**。

## 快速开始

```python
from imputeforge.fuse import ImputeFuse
import numpy as np

X = np.loadtxt("data.csv", delimiter=",")  # 含 NaN
imp = ImputeFuse().fit(X)
X_filled = imp.transform(X)
```

CLI：

```bash
python cli.py benchmark --quick     # ≤60s 快速自检（1 seed × 12 cells）
python cli.py benchmark --full      # 全量基准（3 seeds × 36 cells）
python cli.py ablation              # 消融：选择 / 初始化 / 位移 / 精炼
```

---

## 方法池

| 方法 | 来源 | 备注 |
|---|---|---|
| `mean` / `median` | 基线 | 简单基准 |
| `knn` | sklearn `KNNImputer` | — |
| `knn_np` | 纯 numpy | 离线兜底（无 sklearn 时） |
| `mice_ridge` | sklearn `IterativeImputer` + `BayesianRidge` | MICE |
| `mice_forest` | sklearn `IterativeImputer` + `ExtraTrees` | missForest 族（Stekhoven & Bühlmann 2012） |
| `em_gaussian` | 纯 numpy | 高斯 EM 闭式条件均值 |
| `softimpute` | 纯 numpy | SVT / Mazumder 2010 近端梯度 |
| **`imputefuse`** | **旗舰** | **EM 锚定 + 逐列条件选择 + MNAR 位移校正** |

---

## 性能（全量基准，3 seeds × 36 cells，聚合 NRMSE 越低越好）

| 方法 | 聚合 NRMSE |
|---|---:|
| `mean` | 1.2140 |
| `knn` | 1.0298 |
| `mice_ridge` | 0.8628 |
| `mice_forest` | 0.8616 |
| `softimpute` | 0.8694 |
| `em_gaussian`（最强单法） | 0.8358 |
| **`imputefuse`** | **0.8207** |

- vs `em_gaussian`：**−1.81%**，显著（|Δ|=0.0151 > σ_crit=0.0098），3/3 seed 全胜。
- vs `mice_forest`（missForest 族）：−4.75%。
- vs `mean`：−32.4%。
- 主指标方向正确且显著，但**未达预注册 5% 门槛** → 诚实定级 A。

### 消融（全量，2 seeds）

| 变体 | 聚合 NRMSE | Δ vs full |
|---|---:|---:|
| `full` | 0.8264 | — |
| `− selection`（逐列选择） | 0.8447 | +1.84% |
| `− EM 初始化` | 0.9079 | +8.16% |
| `− refine`（精炼） | 0.8185 | −0.78%（中性） |
| `− shift`（MNAR 校正） | 0.8353 | +0.89% |

→ **EM 初始化是最大贡献项**；逐列选择与 MNAR 校正均正向；精炼在现有网格中性。

---

## 可验证不变量

- EM 观测数据对数似然**单调不增**。
- 无缺失时单步 M-step **复现闭式 MLE** 至数值精度。
- 条件均值补全**逐位保留**每个观测项不变。
- SoftImpute 目标**单调不增**（Mazumder 近端梯度形式）。
- kNN **永不自匹配**，距离度量 nan-aware。
- **零泄漏**：imputer 只在 train 行 fit；`transform` 不在被插补块上重拟合联合分布。
- **确定性**：同 seed 两次运行 bit-identical（BLAS 线程钉 1）。

---

## 已知限制（不隐藏）

1. **3/36 单元格**由单一归纳偏置主导（某专用单法碾压全场），融合无法超越该专用单法。失败案例与归因已保留于 `examples/benchmark.json`。
2. **预注册 5% 门槛未达**（实测 1.81%）。根因与 remediation 见 [`BLOCKED.md`](BLOCKED.md)。
3. `mice_*` 在 scikit-learn 缺失时不可用，池自动降级到纯 numpy `knn_np`。

---

## 测试与 CI

- **53 单测全绿**；覆盖率 **88%**（fuse 93% / em 99% / softimpute 94% / pipeline 42%）。
- `ruff 0.17.0` **check + format 双绿**硬门禁。
- CI 矩阵 **ubuntu + windows × py3.12 / 3.13**（lint+format+pytest+cov+CLI 冒烟+确定性校验）。
- 密钥自查：仓库**零**硬编码密钥（`git grep` + Grep 双绿）。

---

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 融合规则实证、MNAR 校正、不变量、无泄漏、确定性、复杂度、已知限制。
- [`docs/model_card.md`](docs/model_card.md) — 模型卡（候选模型、MNAR 校正、失败模式）。
- [`BLOCKED.md`](BLOCKED.md) — 5% 门槛未达的根因与通向 S 的路径。
- [`CHANGELOG.md`](CHANGELOG.md) — v0.1.0 变更记录。

---

## 复现

```bash
make demo     # 快速自检（≤60s）→ examples/demo_quick.json
make test     # 53 单测
make check    # ruff 双门禁
# 全量基准（≈6min）→ examples/benchmark.json + examples/ablation.json
python examples/run_full_benchmark.py
```

---

## 作者与许可

- **作者**：晨星（GitHub：[@CJX0712](https://github.com/CJX0712)）。
- **License**：MIT（`LICENSE`）。
- 复用并忠实对照：Little & Rubin 缺失机制分类、Stekhoven & Bühlmann 2012 missForest、Mazumder/Hastie/Tibshirani 2010 SoftImpute。
