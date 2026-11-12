"""执行实际DRIE/CFD计算（基础评估+主动学习+优化检查）

入口功能：
- 加载数据集（DRIE或CFD，支持排除ID）
- 运行三阶段：basic（交叉验证）、al（主动学习）、optimization（Pareto检查）
- 记录元数据（配置、代码哈希、版本）
- 支持多进程并行

命令行选项：
--config: 配置文件路径（默认configs/real_main.json）
--run-name: 输出目录名（输出到outputs/results/<run-name>）
--stages: 运行阶段（basic/al/optimization，默认全部）
--exclude-ids: 敏感性分析中排除的实验ID
--workers: 并行进程数（1-4）
--no-optimize: 禁用GP超参数优化（仅数值烟雾测试）

环境配置：
- 临时文件目录设为_tmp（避免系统临时目录污染）
- 限制线程数为1（避免过度订阅）
- 禁止生成.pyc文件
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)

# 环境变量配置
os.environ["TMP"] = str(TEMP)
os.environ["TEMP"] = str(TEMP)
os.environ["MPLCONFIGDIR"] = str(TEMP / "matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# 限制BLAS/MKL线程数，避免过度订阅
for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[env] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

from research.core import GPConfig
from research.data import load_dataset
from research.experiments import (basic_evaluation, run_active_learning, metadata,
                                  write_json, clean_json)
from research.optimization import optimization_readiness


def main():
    """主函数：参数解析、数据加载、三阶段执行"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_main.json")
    parser.add_argument("--run-name", default="real_main")
    parser.add_argument("--stages", nargs="+", choices=["basic", "al", "optimization"],
                        default=["basic", "al", "optimization"])
    parser.add_argument("--exclude-ids", nargs="*", type=int, default=[])
    parser.add_argument("--cfd-variant", choices=["excel", "legacy_code"], default="excel")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-optimize", action="store_true", help="Numerical smoke test only; never label this as the main optimized GPR experiment")
    args = parser.parse_args()
    
    # 验证run-name是简单目录名
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a simple directory name")
    
    # 加载并覆盖配置
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8-sig"))
    for key in ("repeats", "budget"):
        if getattr(args, key) is not None:
            config["experiment"][key] = getattr(args, key)
    for key in ("maxiter", "restarts"):
        if getattr(args, key) is not None:
            config["gp"][key] = getattr(args, key)
    if args.no_optimize:
        config["gp"]["optimize"] = False
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    
    # 初始化配置和数据集
    gp_config = GPConfig(**config["gp"])
    dataset = load_dataset(ROOT, config["dataset"], args.exclude_ids, args.cfd_variant)
    out = ROOT / "outputs" / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    
    config["exclude_ids"] = args.exclude_ids
    config["cfd_variant"] = args.cfd_variant
    
    # 构建元数据记录
    record = metadata(dataset, config, gp_config)
    record["command"] = [sys.executable] + sys.argv
    record["run_name"] = args.run_name
    record["stages_requested"] = args.stages
    record["code_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/"code"/"research").glob("*.py"))}
    record["code_sha256"]["run_research.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    
    # 检查现有运行是否匹配（防止覆盖不同配置的结果）
    existing = out / "run_metadata.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if (old["configuration"] != record["configuration"] or old["source_sha256"] != record["source_sha256"]
                or old.get("code_sha256") != record["code_sha256"]):
            raise RuntimeError("Use a new run-name for changed data/configuration")
    write_json(existing, record)
    
    # 执行三阶段
    started = time.perf_counter()
    basic = None
    try:
        if "basic" in args.stages:
            print(f"{args.run_name}: basic evaluation X={dataset.X.shape}, Y={dataset.Y.shape}", flush=True)
            basic = basic_evaluation(dataset, out, gp_config)
        if "al" in args.stages:
            print(f"{args.run_name}: active learning {config['experiment']}", flush=True)
            run_active_learning(dataset, out, config["experiment"], gp_config, args.workers)
        if "optimization" in args.stages:
            # 从磁盘加载basic结果（如果本次未运行basic阶段）
            if basic is None and (out / "basic_metrics.csv").exists():
                import pandas as pd
                basic = pd.read_csv(out / "basic_metrics.csv")
            optimization_readiness(dataset, out, basic)
        record["status"] = "complete"
        record["stages_completed"] = args.stages
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        # 验证数据源未被修改
        record["source_unchanged"] = hashlib.sha256(Path(dataset.source_path).read_bytes()).hexdigest() == dataset.source_sha256
        write_json(existing, clean_json(record))
    print(f"COMPLETE {out} in {record['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
