"""科学计算入口：验证或重计算（不生成文档）

功能：
- 转发参数到run_submission.py的工作流
- 输出目录重定向到outputs/coding/validation
- 记录此入口的SHA256用于审计

选项：
--recompute-original: 重新计算两个12响应协议（30种子）
--run-prefix: 新输出前缀（避免覆盖保存的结果）
--workers: 并行进程数（1-4）
--comparison-mode: portable（相对容差）或strict（绝对容差）
--reuse-completed: 复用已完成的匹配计算
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import run_submission as workflow

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    """主函数：参数解析与转发"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-original", action="store_true", help="Refit both 12-response protocols with all 30 seeds")
    parser.add_argument("--run-prefix", default="coding_reproduced", help="New output prefix for recomputation")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--comparison-mode", choices=["portable", "strict"], default="portable")
    parser.add_argument("--auc-rtol", type=float, default=workflow.DEFAULT_AUC_RTOL)
    parser.add_argument("--reuse-completed", action="store_true", help="Reuse matching completed prefix results during recomputation")
    args = parser.parse_args(argv)
    
    # 构建转发参数列表
    forwarded = ["--run-prefix", args.run_prefix, "--workers", str(args.workers),
                 "--comparison-mode", args.comparison_mode, "--auc-rtol", str(args.auc_rtol)]
    if args.recompute_original:
        forwarded.append("--recompute-original")
    if args.reuse_completed:
        forwarded.append("--reuse-completed")
    
    # 临时替换sys.argv和输出目录，调用workflow
    previous_args, previous_output = sys.argv, workflow.OUT
    output = ROOT / "outputs/coding/validation"
    try:
        workflow.OUT = output
        sys.argv = [str(ROOT / "run_submission.py"), *forwarded]
        workflow.main()
    finally:
        workflow.OUT, sys.argv = previous_output, previous_args
    
    # 为成功返回的记录添加此入口的标注
    record_path = output / "submission_validation.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["coding_entry_point"] = "run_coding.py"
        record["coding_entry_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record["document_generation_requested"] = False
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
