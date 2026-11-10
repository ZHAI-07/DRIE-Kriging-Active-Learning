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

源码入口：`code/research/data.py`负责数据与来源分组，`core.py`负责GP及采集函数，`experiments.py`负责留出评价和逐轮采样；`code/revision/model_comparison.py`与`early_budget_al.py`保存对照实现。权威输入仍为24×4，六结构输出为24×12。
