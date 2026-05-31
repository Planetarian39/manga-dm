# manga-dm 重构用户故事（优化版）

> 更新日期：2026-05-30  
> 来源：[refactor-plan.md](./refactor-plan.md)  
> 方法：按当前仓库事实拆分 Epic、User Story 和验收标准。当前官方包为 `src`，官方命令为 `manga`。

---

## 全局目标

将已经分层的 `src/` 包继续收敛为稳定的 MaNGA 暗物质分析流水线：

- `src.cli` 只负责参数解析和分发。
- `src.config` 统一管理配置、路径和常量。
- `src.data` 负责数据读取、下载和结果 I/O。
- `src.models` 只保留科学模型和 PyMC 直接辅助函数。
- `src.pipeline` 负责编排 Stage 1、Stage 2 和样本选择。
- `src.stats` 负责模型无关统计工具。
- `src.viz` 负责绘图。
- `src-orig` 只作为历史兼容入口，不承载新开发。

## 全局约束

1. PyMC 核心函数内部不改：`_inf_vel_rot`、`_inf_dm_nfw_pymc`、`fit_m200_c_mcmc` 的先验、似然和采样参数保持不变。
2. 不改变结果 CSV/NetCDF 的文件名、字段名和语义。
3. 不删除 `src-orig/`，除非后续有明确迁移计划。
4. 不把业务逻辑放入 `src.cli`。
5. 新代码不得新增对 `src-orig` 的委托。
6. 每个 Epic 完成后至少运行对应 CLI help 和最小 import smoke test。

---

## Epic 0：校正文档与现状基线

**目标：** 让重构文档反映当前仓库事实，并建立后续验证基线。  
**风险：** 低  
**依赖：** 无

### US-0.1 — 修正包结构描述

**作为** 维护者  
**我想要** 文档明确当前官方实现包是 `src`，命令名是 `manga`  
**以便于** 后续 agent 不会新建错误的 `manga/` 顶层包

**验收标准：**

- [ ] 文档不再要求创建 `manga/` Python 包。
- [ ] 文档明确 `pyproject.toml` 中 `manga = "src.cli.main:main"` 是官方 CLI 入口。
- [ ] 文档明确 `src-orig/` 是历史脚本目录。
- [ ] 文档中的目录职责与 `AGENTS.md` 一致。

### US-0.2 — 建立历史委托清单

**作为** 审阅者  
**我想要** 列出 `src` 中所有运行期委托 `src-orig` 的位置  
**以便于** 后续逐项消除兼容债务

**验收标准：**

- [ ] 使用 `rg "src-orig|import main|import m200|import plates|import figure|import dm" src` 生成清单。
- [ ] 清单按 `pipeline`、`viz`、其他模块分类。
- [ ] 每个委托点标注保留原因和目标替代模块。

### US-0.3 — 建立 CLI 基线验证

**作为** 用户  
**我想要** 确认官方 CLI 与模块入口都能显示帮助  
**以便于** 后续每次重构都能快速发现入口回归

**验收标准：**

- [ ] `python -m src --help` 成功。
- [ ] `manga --help` 成功。
- [ ] `manga select --help` 成功。
- [ ] `manga stage1 --help` 成功。
- [ ] `manga stage2 --help` 成功。
- [ ] `manga figures --help` 成功。
- [ ] `manga merge --help` 成功。
- [ ] `manga sample --help` 成功。

---

## Epic 1：CLI 与配置收敛

**目标：** 让 CLI 全局参数实际影响配置和运行目录。  
**风险：** 中  
**依赖：** Epic 0

### US-1.1 — 接通 `--config`

**作为** 用户  
**我想要** 通过 `manga --config <path>` 指定配置文件  
**以便于** 在不同数据目录和实验配置之间切换

**验收标准：**

- [ ] `src.cli.main` 将 `--config` 传给配置初始化逻辑。
- [ ] `src.config.settings` 支持显式 config path 优先于默认查找。
- [ ] 指定不存在的配置文件时给出清晰错误。
- [ ] `manga --config <path> --help` 不破坏 help 输出。

### US-1.2 — 接通 `--data-dir` 和 `--result-dir`

**作为** 用户  
**我想要** 从 CLI 覆盖数据目录和结果目录  
**以便于** 在不修改 `config.toml` 的情况下运行临时实验

**验收标准：**

- [ ] `--data-dir` 覆盖 `settings.data_dir` 或等效解析结果。
- [ ] `--result-dir` 传递到 `stage1`、`stage2`、`merge`、`sample`、`figures`。
- [ ] 相对路径按项目根目录解析，绝对路径保持不变。
- [ ] `manga stage1 --ifu test --result-dir <path>` 的结果写入指定目录。

### US-1.3 — 保持 CLI 层无业务逻辑

**作为** 开发者  
**我想要** `src.cli` 只做 argparse 和函数分发  
**以便于** pipeline 可以被测试和复用

**验收标准：**

- [ ] `src.cli.main` 不直接读取 FITS、CSV、NetCDF。
- [ ] `src.cli.main` 不直接调用 PyMC 模型。
- [ ] 每个子命令只构造参数并调用 `src.pipeline`、`src.viz` 或 `src.data` 的公开函数。

---

## Epic 2：消除 pipeline 对历史脚本的委托

**目标：** Stage 1、Stage 2、selection 直接运行 `src` 模块中的实现。  
**风险：** 高  
**依赖：** Epic 1

### US-2.1 — Stage 1 直接调用新模块

**作为** 用户  
**我想要** `manga stage1` 不依赖 `src-orig/main.py`  
**以便于** 官方流水线可以在安装包环境中稳定运行

**验收标准：**

- [ ] `src.pipeline.stage1` 不再通过 `sys.path` import `src-orig/main.py`。
- [ ] `process_plate_ifu` 直接调用 `src.models.rotation_curve.RotCurve` 和 `src.models.dm_nfw.DmNfw`。
- [ ] 结果写入通过 `src.data.results` 完成。
- [ ] `manga stage1 --ifu test` 在可用数据环境下输出与历史脚本一致。

### US-2.2 — Stage 2 直接调用新模块

**作为** 用户  
**我想要** `manga stage2 --fit` 不依赖 `src-orig/m200.py`  
**以便于** 群体模型流程由当前包独立维护

**验收标准：**

- [ ] `src.pipeline.stage2` 不再 import `src-orig/m200.py`。
- [ ] `run_stage2(fit=True)` 直接调用 `src.models.population.fit_m200_c_mcmc`。
- [ ] 后验样本加载和合并通过 `src.data.results` 完成。
- [ ] `--quality-cut` 实际影响样本过滤逻辑。

### US-2.3 — 接通 Stage 2 诊断

**作为** 用户  
**我想要** `manga stage2 --diagnose` 运行真实 PSIS 诊断  
**以便于** 不再回退到历史脚本

**验收标准：**

- [x] `--diagnose` 不再打印“not yet wired”。
- [x] 诊断逻辑调用 `src.stats.psis.compute_psis_importance_diagnostics`。
- [x] 缺少必要输入文件时给出清晰错误。

### US-2.4 — selection 直接调用新模块

**作为** 用户  
**我想要** `manga select` 和 `manga sample` 不依赖 `src-orig`  
**以便于** 样本筛选逻辑只维护一份

**验收标准：**

- [ ] `src.pipeline.selection` 不再 import `src-orig/plates.py`。
- [ ] `generate_robustness_sample` 不再 import `src-orig/m200.py`。
- [ ] plateifu 输出路径通过 `settings.resolve_input_path()` 或等效 helper 解析。
- [ ] 旧 `src-orig` 脚本仍可独立运行。

---

## Epic 3：模型层边界清理

**目标：** `src.models` 只保留科学模型和模型直接辅助函数。  
**风险：** 中  
**依赖：** Epic 2

### US-3.1 — 从 `population.py` 移出数据加载逻辑

**作为** 开发者  
**我想要** catalog 和结果读取逻辑从 `src.models.population` 迁出  
**以便于** 模型层不直接承担数据访问职责

**验收标准：**

- [ ] catalog 构建函数迁入 `src.data.catalog`。
- [ ] 后验样本读取函数迁入 `src.data.results`。
- [ ] `src.models.population` 不直接读取项目路径或 CSV 文件，除非是模型输入文件显式参数。

### US-3.2 — 从 `population.py` 移出样本筛选逻辑

**作为** 开发者  
**我想要** 样本过滤和鲁棒性子样本逻辑归入 `src.pipeline.selection`  
**以便于** 模型层只消费已准备好的输入

**验收标准：**

- [ ] `_filter_dataframe_by_*` 和 `_resolve_quality_filter_thresholds` 位于 `src.pipeline.selection`。
- [ ] `generate_robustness_sample` 位于 `src.pipeline.selection`。
- [ ] `src.models.population.fit_m200_c_mcmc` 接收准备好的模型输入或通过明确参数加载。

### US-3.3 — 从 `population.py` 移出绘图逻辑

**作为** 开发者  
**我想要** 绘图函数归入 `src.viz`  
**以便于** 模型层不依赖 matplotlib 图形职责

**验收标准：**

- [ ] 群体关系图函数位于 `src.viz.paper`。
- [ ] 后验诊断图函数位于 `src.viz.posterior`。
- [ ] `src.models.population` 不定义 `plot_*` 函数。

---

## Epic 4：消除 viz 对历史脚本的委托

**目标：** 所有绘图逻辑直接位于 `src.viz`。  
**风险：** 中  
**依赖：** Epic 3

### US-4.1 — 实现 RC 曲线图

**作为** 用户  
**我想要** `src.viz.rc_curves` 直接生成 RC 图  
**以便于** 不再依赖 `src-orig/figure.py`

**验收标准：**

- [ ] `src.viz.rc_curves` 不再 import `src-orig/figure.py`。
- [ ] `plot_rc_fit_summary_comparison` 和 `plot_rc_fit_summary_panels` 保持原有输出路径和参数语义。
- [ ] 至少一个指定 IFU 的图文件可生成且非空。

### US-4.2 — 实现速度场图

**作为** 用户  
**我想要** `src.viz.velocity_maps` 直接生成速度场图  
**以便于** 速度场可视化可独立维护

**验收标准：**

- [ ] `src.viz.velocity_maps` 不再 import `src-orig/figure.py`。
- [ ] 函数通过 `src.data.maps` 获取速度场数据。
- [ ] 输出图文件存在且非空。

### US-4.3 — 实现后验诊断图

**作为** 用户  
**我想要** `src.viz.posterior` 直接生成后验诊断图和 pair plot 注解  
**以便于** 后验可视化不再依赖历史脚本

**验收标准：**

- [ ] `src.viz.posterior` 不再 import `src-orig/dm.py` 或 `src-orig/m200.py`。
- [ ] pair plot 标签和区间注解使用 `src.stats.intervals` 的统一实现。
- [ ] 后验诊断图输出文件存在且非空。

### US-4.4 — 实现论文组合图

**作为** 用户  
**我想要** `src.viz.paper` 直接生成论文级组合图  
**以便于** `manga figures` 可以完全使用当前包

**验收标准：**

- [ ] `src.viz.paper` 不再 import `src-orig/figure.py` 或 `src-orig/m200.py`。
- [ ] `GalaxyFigureData` 和数据准备逻辑位于 `src.viz.paper` 或更合适的 `src.viz` 内部模块。
- [ ] `manga figures --ifu <plateifu>` 可生成预期图文件。

---

## Epic 5：兼容层与文档收尾

**目标：** 明确 `src-orig` 长期定位，避免新旧实现继续分叉。  
**风险：** 低  
**依赖：** Epic 4

### US-5.1 — 标记历史脚本边界

**作为** 维护者  
**我想要** `src-orig` 明确标记为 legacy  
**以便于** 后续修改不会继续向历史脚本添加新功能

**验收标准：**

- [x] README 说明 `src-orig` 仅用于兼容旧调用。
- [x] AGENTS 或重构文档说明新功能必须进入 `src`。
- [x] `src-orig` 不被当前 `src` 运行期路径依赖，或仅剩明确批准的兼容点。

### US-5.2 — 同步 README 命令示例

**作为** 用户  
**我想要** README 中的命令与实际 CLI 一致  
**以便于** 按文档可以成功运行

**验收标准：**

- [x] README 包含 `python -m src` 和 `manga --help`。
- [x] README 示例覆盖 `select`、`stage1`、`stage2`、`figures`、`merge`、`sample`。
- [x] README 明确数据缺失时哪些 smoke test 仍可运行。

### US-5.3 — 最终依赖扫描

**作为** 审阅者  
**我想要** 确认当前包不再运行期依赖历史脚本  
**以便于** 完成重构收尾

**验收标准：**

- [ ] `rg "src-orig|import main|import m200|import plates|import figure|import dm" src` 没有未批准的运行期委托。
- [ ] `python -m src --help` 成功。
- [ ] `manga --help` 成功。
- [ ] 所有子命令 help 成功。
- [ ] 如果完整科学验证因缺少数据未运行，文档记录具体原因。

---

## 依赖关系

```text
Epic 0 文档与基线
  └── Epic 1 CLI 与配置
        └── Epic 2 pipeline 去历史委托
              └── Epic 3 模型层边界清理
                    └── Epic 4 viz 去历史委托
                          └── Epic 5 兼容层与文档收尾
```

---

## Definition of Done

一个 User Story 完成必须满足：

1. 文档、代码或验证命令与当前 `src` 包结构一致。
2. 未新增对 `src-orig` 的运行期依赖。
3. CLI help 和相关子命令 help 通过。
4. 涉及 pipeline、models、viz 的变更有最小 smoke test 或明确说明未运行原因。
5. 涉及数值输出的变更有迁移前后对比，或说明数据/算力阻塞点。

---

## 进度追踪

| Epic | Story | 状态 |
|------|-------|------|
| Epic 0 | US-0.1 修正包结构描述 | completed |
| Epic 0 | US-0.2 建立历史委托清单 | completed |
| Epic 0 | US-0.3 建立 CLI 基线验证 | completed |
| Epic 1 | US-1.1 接通 `--config` | completed |
| Epic 1 | US-1.2 接通目录 override | completed |
| Epic 1 | US-1.3 保持 CLI 无业务逻辑 | completed |
| Epic 2 | US-2.1 Stage 1 去历史委托 | completed |
| Epic 2 | US-2.2 Stage 2 去历史委托 | completed |
| Epic 2 | US-2.3 接通 Stage 2 诊断 | completed |
| Epic 2 | US-2.4 selection 去历史委托 | completed |
| Epic 3 | US-3.1 移出数据加载逻辑 | pending |
| Epic 3 | US-3.2 移出样本筛选逻辑 | pending |
| Epic 3 | US-3.3 移出绘图逻辑 | pending |
| Epic 4 | US-4.1 实现 RC 曲线图 | completed |
| Epic 4 | US-4.2 实现速度场图 | completed |
| Epic 4 | US-4.3 实现后验诊断图 | completed |
| Epic 4 | US-4.4 实现论文组合图 | completed |
| Epic 5 | US-5.1 标记历史脚本边界 | completed |
| Epic 5 | US-5.2 同步 README 命令示例 | completed |
| Epic 5 | US-5.3 最终依赖扫描 | completed |
