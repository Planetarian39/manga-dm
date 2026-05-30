# manga-dm 重构方案审计与优化计划

> 更新日期：2026-05-30  
> 审核对象：当前仓库实现、`pyproject.toml`、`README.md`、`src/`、`src-orig/`  
> 结论：原计划的分层方向正确，但目标包结构已经过期。当前官方实现位于 `src/` 包，历史脚本保留在 `src-orig/`，后续重构应围绕“减少 `src` 对 `src-orig` 的运行期依赖、补齐 CLI 与验证闭环”推进。

---

## 1. 审计结论

### 1.1 已完成的结构调整

当前仓库已经具备以下目标结构：

```text
src/
├── __main__.py
├── cli/
├── config/
├── data/
├── models/
├── pipeline/
├── stats/
└── viz/

src-orig/
├── main.py
├── rc.py
├── dm.py
├── m200.py
├── figure.py
├── plates.py
└── util/
```

`pyproject.toml` 已注册官方 CLI：

```toml
[project.scripts]
manga = "src.cli.main:main"
```

因此后续文档不应再规划新建 `manga/` 顶层包，也不应把 `src/` 描述为即将删除的过渡层。当前事实是：

- `src/` 是当前主实现包。
- `src-orig/` 是历史兼容脚本集合。
- `manga` 是安装后的命令名，不是当前 Python 包名。
- `python -m src` 是模块入口。

### 1.2 原方案中需要修正的问题

1. **包名过期**  
   原方案要求新建 `manga/` 包，但仓库已经使用 `src` 包并通过 `manga = "src.cli.main:main"` 暴露命令。继续按原计划执行会制造第二套包结构。

2. **兼容层方向写反**  
   原方案写成 `src/*.py` 未来作为 wrapper。当前历史入口在 `src-orig/`，主实现在 `src/`。优化后应要求：保留 `src-orig/` 可运行，逐步减少 `src` 对 `src-orig` 的委托。

3. **完成度被高估**  
   虽然目录已分层，但 `src.pipeline.stage1`、`src.pipeline.stage2`、`src.pipeline.selection`、`src.viz.*` 中仍存在对 `src-orig` 的委托或占位逻辑。结构迁移已开始，但业务逻辑尚未完全收敛到 `src/`。

4. **依赖规则需要允许历史隔离例外**  
   目标规则仍应禁止 `src` 的业务模块反向依赖 CLI 或跨层依赖，但短期内允许明确标记的 `src-orig` 兼容委托。每个委托点必须列入清单并逐步消除。

5. **验证要求需要前移**  
   原方案到模型迁移阶段才强调数值一致。当前已经存在新旧双路径，应先建立基线对比命令，再继续替换委托逻辑。

---

## 2. 优化后的目标架构

### 2.1 官方包结构

```text
manga-dm/
├── pyproject.toml
├── README.md
├── src/                         # 当前官方实现包
│   ├── __init__.py
│   ├── __main__.py              # python -m src
│   ├── cli/                     # 只做参数解析和命令分发
│   ├── config/                  # 配置、常量、路径解析
│   ├── data/                    # 外部数据和结果文件 I/O
│   ├── models/                  # 科学模型与 PyMC 实现
│   ├── pipeline/                # Stage 1/2 工作流编排
│   ├── stats/                   # 模型无关统计工具
│   └── viz/                     # 绘图和论文图生成
├── src-orig/                    # 历史兼容入口，不承载新开发
└── docs/refactor/
```

### 2.2 层级依赖规则

| 层 | 允许依赖 |
|----|---------|
| `src.cli` | `src.pipeline`、`src.viz`、`src.data`、`src.config` |
| `src.pipeline` | `src.models`、`src.data`、`src.stats`、`src.config` |
| `src.viz` | `src.models`、`src.data`、`src.stats`、`src.config` |
| `src.models` | `src.data`、`src.stats`、`src.config` |
| `src.data` | `src.config` 和第三方库 |
| `src.stats` | `src.config` 和第三方库 |
| `src.config` | 标准库和第三方库 |

禁止规则：

- `src.data` 不得 import `src.models`、`src.pipeline`、`src.viz`。
- `src.stats` 不得 import `src.models`、`src.pipeline`、`src.viz`。
- `src.models` 不得 import `src.pipeline`、`src.viz`、`src.cli`。
- `src.viz` 不得 import `src.pipeline`、`src.cli`。
- `src.config` 不得 import 任何业务层。
- 新代码不得新增对 `src-orig` 的依赖；已有委托点只能在清单内保留，并在后续阶段移除。

---

## 3. 当前剩余风险清单

| 风险 | 当前表现 | 影响 | 优先级 |
|------|---------|------|--------|
| 运行期委托历史脚本 | `src.pipeline.*`、`src.viz.*` 仍通过 `sys.path` import `src-orig` | 新架构边界不稳定，打包后行为依赖源码布局 | 高 |
| CLI 参数未完全贯通 | 顶层参数 `--config`、`--data-dir`、`--result-dir`、`--verbose` 已解析但未统一初始化 settings | 用户传参可能无效或表现不一致 | 高 |
| Stage 2 诊断未接线 | `--diagnose` 打印未接线提示 | CLI 功能表面存在，实际不可用 | 中 |
| 可视化层多处 wrapper | `src.viz.*` 仍委托 `src-orig/figure.py` 或 `src-orig/m200.py` | 图生成难以单独验证和维护 | 中 |
| 模型与 pipeline 归属不完全一致 | `src.models.population` 中仍包含部分 catalog、selection、plotting、pipeline 逻辑 | 层级边界被污染 | 中 |
| 验证基线缺失 | 缺少固定的新旧路径对比脚本或最小 smoke test 清单 | 难以证明“重构无行为回归” | 高 |

---

## 4. 优化后的实施阶段

### Phase 0：冻结事实与建立基线

目标：确认当前 `src` 是官方实现包，并为后续迁移建立可重复验证基线。

交付：

- 更新重构文档，废弃“新建 `manga/` 包”的描述。
- 记录 `src` 对 `src-orig` 的所有委托点。
- 建立最小验证命令清单：
  - `python -m src --help`
  - `manga --help`
  - `manga select --help`
  - `manga stage1 --help`
  - `manga stage2 --help`
  - `manga figures --help`
  - `manga merge --help`
  - `manga sample --help`

验收：

- 文档中的包名、目录职责、入口点与当前仓库一致。
- 所有 CLI help 命令可运行。

### Phase 1：CLI 与 settings 收敛

目标：让 CLI 全局参数真正影响运行配置，避免解析后丢弃。

交付：

- `src.cli.main` 将 `--config`、`--data-dir`、`--result-dir` 传递到配置初始化或具体 pipeline 调用。
- `src.config.settings` 明确配置查找顺序和 CLI override 优先级。
- 所有子命令 help 保持可用。

验收：

- `manga --result-dir <path> stage1 --ifu test` 使用指定结果目录。
- `manga --config <path> ...` 使用指定配置文件。
- README 与文档中的命令示例一致。

### Phase 2：消除 pipeline 对 `src-orig` 的委托

目标：Stage 1、Stage 2、selection 的运行逻辑直接使用 `src` 模块。

交付：

- `src.pipeline.stage1` 不再 import `src-orig/main.py`。
- `src.pipeline.stage2` 不再 import `src-orig/m200.py`。
- `src.pipeline.selection` 不再 import `src-orig/plates.py` 或 `src-orig/m200.py`。
- 保留 `src-orig` 脚本自身可运行，不反向依赖新 CLI。

验收：

- `rg "src-orig|import main|import m200|import plates" src/pipeline` 无新增委托。
- `manga stage1 --ifu test` 与历史脚本在最小样本上输出一致。
- `manga stage2 --fit` 至少可在最小可行输入上启动并写出预期结果。

### Phase 3：清理模型层边界

目标：让 `src.models` 只保留科学模型和模型直接辅助函数。

交付：

- 从 `src.models.population` 移出 catalog 加载、样本选择、结果读取、绘图函数。
- catalog 相关逻辑归入 `src.data.catalog`。
- 样本筛选和鲁棒性子样本逻辑归入 `src.pipeline.selection`。
- 结果 I/O 归入 `src.data.results`。
- 群体关系绘图归入 `src.viz.paper` 或 `src.viz.posterior`。

验收：

- `src.models.population` 不再包含 plot、catalog loading、pipeline orchestration 逻辑。
- PyMC 核心函数 `fit_m200_c_mcmc` 的函数体保持不变。
- 最小 Stage 2 验证通过。

### Phase 4：消除 viz 对 `src-orig` 的委托

目标：可视化逻辑完全位于 `src.viz`。

交付：

- `src.viz.rc_curves` 直接实现 RC 汇总图。
- `src.viz.velocity_maps` 直接实现速度场图。
- `src.viz.posterior` 直接实现后验诊断和 pair plot 注解。
- `src.viz.paper` 直接实现论文组合图。
- `src.viz.utils` 不再委托 `src-orig/util/plot_util.py`。

验收：

- `rg "src-orig|import figure|import m200|import dm" src/viz` 无历史委托。
- `manga figures --ifu <plateifu>` 能生成图文件。
- 迁移前后同一输入生成的图文件存在且核心面板数量一致。

### Phase 5：兼容层收尾与文档同步

目标：明确 `src-orig` 的长期定位，避免未来修改继续分叉。

交付：

- `README.md`、`AGENTS.md`、重构文档对入口点描述一致。
- `src-orig` 标记为 legacy，只接受兼容性修复。
- 新功能只能进入 `src` 分层模块。
- 建立最终验收清单和进度表。

验收：

- `manga --help` 和所有子命令 help 可用。
- `python -m src --help` 可用。
- 历史脚本仍可运行或文档明确说明其兼容范围。
- `rg "src-orig" src` 只剩允许的兼容说明，或完全清零。

---

## 5. 不变约束

- 不修改 PyMC 核心模型内部：`_inf_vel_rot`、`_inf_dm_nfw_pymc`、`fit_m200_c_mcmc` 的先验、似然、采样参数保持不变。
- 不改变结果 CSV/NetCDF 的文件名、字段名和数值语义。
- 不移除 `src-orig/` 历史入口，除非有明确迁移计划和用户确认。
- 不新增全局状态或隐藏副作用。
- 不把业务逻辑放入 `src.cli`。
- 不为了重构引入新依赖，除非现有依赖无法覆盖需求。

---

## 6. 验证策略

### 6.1 每次文档或 CLI 变更

```bash
python -m src --help
manga --help
manga select --help
manga stage1 --help
manga stage2 --help
manga figures --help
manga merge --help
manga sample --help
```

### 6.2 pipeline 或 data 变更

```bash
manga stage1 --ifu test
manga merge --ifu-file <existing-plateifu-file>
```

如果本地缺少数据，记录未运行原因，并至少运行 help 与 import smoke test：

```bash
python -c "import src; import src.cli.main; import src.config.settings"
```

### 6.3 models 变更

- 对固定 `TEST_PLATE_IFUS` 运行迁移前后对比。
- 比较 `rc_param.csv`、`nfw_param_cm200.csv`、后验 NetCDF 中关键变量。
- 如果 MCMC 具有随机性，固定随机种子或使用统计容差，而不是宣称 bit-for-bit。

### 6.4 viz 变更

- 生成至少一个指定 IFU 的图文件。
- 检查输出文件存在、大小非零。
- 对关键图形保留迁移前后样例以便人工对照。

---

## 7. 推荐执行顺序

1. 先完成 Phase 0，确保文档与现实一致。
2. 然后做 Phase 1，修正 CLI 参数实际生效问题。
3. 再做 Phase 2，优先消除 pipeline 对 `src-orig` 的委托。
4. 接着做 Phase 3，把 `src.models.population` 中的非模型职责迁出。
5. 最后做 Phase 4 和 Phase 5，收敛可视化委托和兼容说明。

这个顺序优先处理用户可见入口和验证基线，再处理内部边界，风险低于继续按原 Wave 计划迁移目录。
