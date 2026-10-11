# SOP §11 回写 · delivered.md

> 由 ImputeForge 交付流程自动/半自动回填。域 / 日期 / tag / 关键指标（含门槛与是否通过）。

---

## ImputeForge

| 字段 | 值 |
|---|---|
| 系统 | **ImputeForge** |
| 域 | 表格缺失值插补 Tabular Missing-Value Imputation（MCAR / MAR / MNAR） |
| 日期 | 2026-10-11（v0.1.0） |
| tag | `v0.1.0` |
| 仓库 | https://github.com/CJX0712/imputeforge |
| 作者 | 晨星 |

### 关键指标（全量基准，3 seeds × 36 cells）

| 指标 | 实测 | 门槛 | 通过 |
|---|---|---|---|
| 聚合 NRMSE `imputefuse` vs 最强单法 `em_gaussian` | **−1.81%（显著）** | ≥ 5% | ❌ |
| vs missForest 族 `mice_forest` | −4.75% | — | ✅ |
| vs 均值插补 `mean` | −32.4% | — | ✅ |
| 显著性 | \|Δ\|=0.0151 > σ_crit=0.0098，3/3 seed 全胜 | — | ✅ |

### 工程质量门禁

| 项 | 值 | 通过 |
|---|---|---|
| 单测 | 53 / 53 全绿 | ✅ |
| 覆盖率 | 88%（fuse 93% / em 99% / softimpute 94%） | ✅ |
| ruff 0.17.0 check+format | 双绿 | ✅ |
| CI 矩阵 | ubuntu+windows × py3.12/3.13 | ✅ |
| 密钥自查 | 零硬编码（`git grep`+Grep 双绿） | ✅ |
| 确定性 | bit-identical（同 seed 两次运行） | ✅ |
| 零泄漏 | 仅在 train fit，transform 不重拟合 | ✅ |

### 质量等级

**A（诚实，非 S）** —— 主指标方向正确且显著，但未达预注册 5% 门槛；如实标注于 README / BLOCKED.md，未将 A 标为 S 推送。

### 消融主导项

EM 初始化（移除 +8.16%）> 逐列选择（+1.84%）> MNAR 位移（+0.89%）> refine（中性）。

### 已知限制（不隐藏）

3/36 单元格由单一归纳偏置主导，融合无法超越专用单法；失败案例保留于 `examples/benchmark.json`。
