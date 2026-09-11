# 项目目标

**RefDiff** is a *Reference Restoration Diffusion* framework built upon the principle of *Waveform Refinement Diffusion*, progressively transforming impaired received signals toward their clean reference waveforms to enable robust modulation recognition or other utilization.

从受损复基带信号中剥离信道和接收机损伤，恢复只保留信号本身特征的表示。当前 baseline 用条件 1D diffusion 从接收频谱恢复干净发射频谱。



# Python 环境

- 使用 Conda 环境 `rfsig`；非交互命令使用 `conda run -n rfsig <command>`。
- 未经用户明确同意，不安装或升级依赖。



# 项目组织规范

## 总体原则

- 仓库根目录只保留全局工程文件和职责明确的一级目录，不在根目录堆放临时脚本、模型权重或实验图片。
- 仓库只使用根目录的一份 `pyproject.toml`。`rid2026`、`refdiff` 和 `amc` 是三个独立的 Python 包，但在当前研究阶段由同一工程统一安装和管理。
- 配置文件必须就近放在使用它的测试或实验中，不设置全局 `configs/` 目录。
- 只有可复用的业务逻辑才进入 `src/`；测试组装代码属于 `tests/`，实验组装代码属于 `experiments/`。

## 顶层目录职责

```text
.
├── src/          Python 包源码
├── tests/        验证代码正确性的编号测试事项
├── experiments/  验证研究假设的编号实验事项
├── datasets/     RID2026、RML2016A 等本地数据集
├── docs/         项目设计、数据契约和研究协议文档
├── pictures/     docs 引用的架构图、流程图等说明性图片
├── references/   只读的论文和外部参考实现
└── archive/      不再生效但需要保留的历史方案
```

### `src/`

- 一级子目录为可导入的包，当前主要包为 `src/rid2026/`、`src/refdiff/` 和 `src/amc/`。
- 包目录内的 `README.md` 后直接放 `__init__.py`、模块文件或按职责细分的文件夹；不再嵌套第二层 `src/`，也不放包级 `pyproject.toml`。
- `rid2026` 拥有波形、损伤、样本生成和数据落盘逻辑；`refdiff` 拥有恢复数据读取、diffusion 模型、训练和恢复指标；`amc` 拥有调制分类的模型、训练和评估逻辑。
- 依赖边界为：`rid2026` 不依赖 `refdiff` 或 `amc`；`refdiff` 不依赖 `amc`；`amc` 不依赖 `refdiff`。需要组合多个包的流程放入 `tests/` 或 `experiments/`。

### RefDiff 网络版本管理

- 当前原始 RefDiff 网络保留为 `refdiff`，不在名称中添加版本号；它是唯一不带版本号的 RefDiff 网络版本。
- 任何网络结构修改都必须先完整复制当前版本，再在副本上修改；禁止直接覆盖或改写已有版本的网络结构。
- 后续版本使用 `refdiff_<主版本>_<小版本>` 命名。每次网络结构修改只递增小版本号，例如 `refdiff_1_1`、`refdiff_1_2`、`refdiff_1_3`。
- 每个版本必须保持代码、训练配置、实验输出和 checkpoint 的版本对应关系，不能用一个版本的代码加载或覆盖另一个版本的权重。
- 权重文件名必须包含模型版本号，例如 `refdiff_1_1_best.pt`、`refdiff_1_1_last.pt`；原始无版本号网络的权重可以保留为 `refdiff_best.pt`、`refdiff_last.pt`。
- 版本号一旦用于代码、实验结果或 checkpoint，就不得复用、重命名或改变其含义；新结构只能创建新的版本号。

### `tests/`

- `tests/` 用于判断代码能否正常运行、数值实现是否正确、接口和包边界是否被遵守；不用它证明研究效果。
- 一个独立测试事项对应一个目录，命名为 `NN_<topic>/`，例如 `00_rid2026_waveforms/`、`01_rid2026_impairments/` 或 `05_phase1_smoke_pipeline/`。
- `NN` 使用两位数字并按建立顺序递增。编号一旦被文档、提交或结果引用就不再重排；删除事项后可保留编号空缺。
- 每个事项放自己的 `test_*.py`。需要说明复杂前置或判定标准时添加 `README.md`；需要固定输入时添加 `fixtures/`。
- 单个测试配置直接命名为 `config.yaml`；只有同一事项需要多套配置时才建立 `configs/`。
- 测试产生的临时文件应使用 pytest 的临时目录，不在 `tests/` 内长期保存 checkpoint、日志或实验图片。

## `tests/` 事项示例

```text
tests/
├── 00_rid2026_waveforms/
│   ├── README.md
│   └── test_waveforms.py
├── 01_rid2026_generation/
│   ├── config.yaml
│   ├── fixtures/
│   └── test_generation.py
└── 02_phase1_smoke_pipeline/
    └── test_pipeline.py
```

上述只是形式示例，不是预先固定的测试清单。应根据实际需要新建、合并或停用测试事项。

### `experiments/`

- `experiments/` 用于回答具体研究问题，例如恢复后 NMSE 是否降低，或固定 AMC 上的分类准确率是否提升。
- 一个独立实验对应一个目录，命名为 `NN_<topic>/`，例如 `00_generate_rid2026/`、`01_train_clean_amc/` 或 `04_evaluate_amc_improvement/`。
- 实验编号遵守与测试编号相同的稳定性规则：只追加，不因目录排序、删除或重命而重新编号。
- 每个实验通常包含 `README.md`、`run.py`、配置和 `outputs/`。`README.md` 说明研究问题、变量、输入、评估指标和执行方法；`run.py` 只组装 `src/` 中的可复用能力。
- 单套配置使用 `config.yaml`；多套配置放入 `configs/`。每次运行必须以开始时间创建独立目录 `outputs/YYYYMMDD_HHMMSS/`；同一秒内重复启动时应添加递增后缀，避免覆盖已有运行。
- 每次运行的标准输出和标准错误必须在终端实时显示的同时完整写入 `outputs/<run_time>/run.log`，包括训练进度、指标和完整 traceback，以便定位和复现报错。
- 本次运行的实际配置、指标历史、checkpoint、恢复样本和结果图应与 `run.log` 保存在同一 `outputs/<run_time>/` 目录，使每次运行可独立追溯。如需长期保留经整理的最终指标或图表，可另建 `results/`，并在实验 `README.md` 中说明来源运行。
- `outputs/` 与其他大型或可重生的实验产物应由 `.gitignore` 排除；需要纳入版本管理的摘要指标或最终图表应在实验 `README.md` 中明确说明。

## `experiments/` 事项示例

```text
experiments/
├── 00_generate_rid2026/
│   ├── README.md
│   ├── configs/
│   │   ├── smoke.yaml
│   │   └── full.yaml
│   ├── run.py
│   └── outputs/
└── 01_train_refdiff_baseline/
    ├── README.md
    ├── config.yaml
    ├── run.py
    └── outputs/
        └── 20260910_204202/
            ├── run.log
            ├── config.json
            └── history.json
```

上述同样只是形式示例。实验目录应随研究问题演进，不应按预设清单创建空目录。

### `datasets/`

- 每个数据集使用独立子目录，例如 `datasets/rid2026/` 或 `datasets/rml2016a/`。
- `datasets/README.md` 记录数据来源、生成或获取方式、磁盘格式和预计容量。
- 大型数据、分片和中间缓存不提交 Git；只跟踪必要的 README 或小型格式示例。

### `docs/` 与 `pictures/`

- `docs/` 保存由 AI 和开发者共同维护的设计与研究文档。文档应描述当前有效状态，不把历史方案混入当前设计。
- `pictures/` 保存 `docs/` 中引用的说明性图片，并按主题划分子目录。实验直接产生的曲线、混淆矩阵和恢复样本应保留在对应实验的 `results/` 中。
