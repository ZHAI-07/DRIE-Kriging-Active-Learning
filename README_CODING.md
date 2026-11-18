# DRIE科学计算代码

当前工作先处理数据、模型、主动学习、复算与测试。报告、PPT和图表按需要另行启动。

推荐Python3.12。科学运行及测试依赖见`requirements_coding.txt`，无需安装python-docx、PowerPoint库或专用文档运行时。

本仓库保留运行所需的数据快照、保存结果和测试夹具，克隆后可直接校验。课程报告、PPT、文档生成模块、虚拟环境和临时文件不纳入仓库；使用`run_coding.py`作为入口。

```powershell
python -m pip install -r requirements_coding.txt
# 默认只校验已有科学结果并运行测试
python -B run_coding.py
# 两种数据情景各30种子完整重算
python -B run_coding.py --recompute-original --run-prefix coding_run --workers 4
# 复用已完成且来源、配置和核心代码匹配的计算
python -B run_coding.py --recompute-original --run-prefix coding_run --reuse-completed
```

默认比较要求选点完全一致，逐行AUC相对误差不超过3%；曲线偏差另作诊断。`--comparison-mode strict`使用绝对1e-8的数值检查。验收标准详见`run_submission.py`，两入口共用同一实现。

模型对照和小预算主动学习的重算：

```powershell
python -B run_revision.py --recompute --workers 4
```

这条命令只进行科学计算和校验。代码入口不自动生成报告、PPT或图表。验证记录位于`outputs/coding/validation/`及各计算模块的结果目录。

包中另保留科研绘图源码；需要绘图时可按`requirements.lock.txt`安装相应科学绘图库，默认校验不需要它们。内部兼容入口中的文档选项不属于本科学代码仓库的使用范围。

源码入口：`code/research/data.py`负责数据与来源分组，`core.py`负责GP及采集函数，`experiments.py`负责留出评价和逐轮采样；`protocol.py`提供小样本评估协议（按设计格留出、噪声上限、行置换对照、伪重复检查），模型比较应统一经它报告；`code/revision/model_comparison.py`与`early_budget_al.py`保存对照实现。权威输入仍为24×4，六结构输出为24×12。

小样本评估协议的口径（`code/research/protocol.py`，测试见`tests/test_evaluation_protocol.py`）：23条可用记录只占15个(Coil, Platen)设计格，5格含重复（4/3/2/2/2条），因此留出单元必须是**格**而非记录——按记录留出会让同格重复样本留在训练集，实测同一模型因此从R²+0.209虚高到+0.410。格内差异为测量噪声，可达R²上限为`1 - SS_within/SS_total`（CD 0.900、AR 0.868、全体 0.884），报告R²时应同时给出该上限。凡"加入某些输入带来提升"的结论必须附行置换对照，以区分正则化几何效应与真实信息；配对检验前必须先数独立初始设计单元，固定设计下重复记录选择不构成独立重复。运行`python -B -c "import sys;sys.path.insert(0,'code');from research.protocol import main;main()"`可打印当前数据的完整设计审计。
