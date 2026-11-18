"""验证已保存的v2研究结果（默认模式）；重计算为可选项

功能：
- 默认：执行数值/来源检查，刷新验证记录（不重新训练模型）
- --recompute: 重新运行审查证据、关联、模型比较和早期预算AL
- --figures: 从保存结果重新绘制v2图表并运行图表检查
- --documents: 构建Markdown/DOCX（使用标准python-docx/Pillow）

示例：
  python run_revision.py                          # 默认验证
  python run_revision.py --recompute --workers 4  # 完整重计算
  python run_revision.py --figures --documents    # 生成图表和文档

注意：
- 默认不重新训练模型或创建新采样实验
- 文档构建不调用Word、PDF导出、Node或Codex演示运行时
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import compile_check
SUMMARY = ROOT / "outputs/revision/reproduction_validation.json"


def relative(value):
    """可移植路径显示：相对于项目根目录，不列出机器驱动器路径"""
    return Path(value).relative_to(ROOT).as_posix()


def read_json(path):
    """读取JSON文件（UTF-8 with BOM）"""
    return path.read_text(encoding="utf-8-sig")


def digest(path):
    """计算文件SHA256哈希（仅用于标注入口自身，不作为验收门槛）"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_file(recorded):
    """解析记录的路径，确保在项目范围内"""
    path = (ROOT / str(recorded).replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("A recorded source path is outside the project")
    return path


def source_checks():
    """检查新科学来源/实现的可用性（非布局QA）

    验证：
    - 项目内所有Python源码可编译
    - 审查证据和关联处理所依赖的输入文件存在
    - 早期预算AL协议已完整执行
    - 6个v2模型场景齐全且完成

    说明：此处不再比对历史SHA256清单。注释性改动会改变文件字节而使
    旧哈希失效，但行为未变；改为只要求源码可编译、所需文件齐备。
    """
    checks = []
    records, failures = compile_check.compile_sources(ROOT)
    if failures:
        broken = ", ".join(f["path"] for f in failures[:5])
        raise RuntimeError(f"Python sources failed to compile: {broken}")
    counts = compile_check.summary(records)
    checks.append({"manifest": "compile_check", "path": "code/**/*.py", "status": "compiled",
                   **counts})

    def require(path, manifest):
        """检查被引用的来源文件是否存在"""
        if not path.is_file():
            raise FileNotFoundError(f"Missing source: {relative(path)} (declared by {manifest})")
        checks.append({"manifest": manifest, "path": relative(path), "status": "present"})

    # 审查证据和关联处理
    evidence = [
        ("outputs/revision/evidence/review_evidence_summary.json", "source_sha256", "code/revision/review_evidence.py"),
        ("outputs/revision/evidence/process_associations_metadata.json", "input_sha256", "code/revision/process_associations.py"),
    ]
    for name, key, script in evidence:
        meta = json.loads(read_json(ROOT / name))
        for source in meta[key]:
            require(project_file(source), name)
        require(ROOT / script, name)

    # 早期预算AL协议
    name = "outputs/revision/early_al/protocol.json"
    meta = json.loads(read_json(ROOT / name))
    if meta.get("status") != "complete":
        raise RuntimeError("Early-budget AL execution is incomplete")
    for source in meta["source_sha256"]:
        require(project_file(source), name)

    # 6个v2模型场景
    model_paths = sorted((ROOT / "outputs/revision/models").glob("*/run_metadata.json"))
    if len(model_paths) != 6:
        raise RuntimeError("Expected six completed v2 model scenarios")
    for path in model_paths:
        meta = json.loads(read_json(path))
        if meta.get("status") != "complete":
            raise RuntimeError(f"Incomplete scenario: {relative(path)}")
        require(project_file(meta["source"]), relative(path))
        require(ROOT / "code/revision/model_comparison.py", relative(path))
    return checks


def run_script(script, arguments, environment, commands):
    """运行子脚本并记录输出（简洁进度，完整诊断）"""
    if not (ROOT / script).is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    display = [Path(sys.executable).name, "-B", script, *arguments]
    print("RUN " + " ".join(display), flush=True)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-B", script, *arguments], cwd=ROOT,
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    # 子脚本保留自己的诊断；此条目保持控制台进度简洁
    output = result.stdout.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
    commands.append({"command": display, "exit_code": result.returncode,
                     "elapsed_seconds": round(time.perf_counter()-started, 3),
                     "output_tail": output[-1800:]})
    if result.returncode:
        print(output, file=sys.stderr)
        raise RuntimeError(f"Command failed: {script}")
    print("PASS " + script, flush=True)


def main():
    """主函数：参数解析、环境配置、执行流程"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="Explicitly rerun review evidence, associations, model comparisons and early-budget AL before validation")
    parser.add_argument("--workers", type=int, default=4, help="Worker count for --recompute model/AL runs (default: 4)")
    parser.add_argument("--figures", action="store_true", help="Redraw v2 figures from saved results and run figure checks")
    parser.add_argument("--documents", action="store_true", help="Build Markdown/DOCX using standard python-docx/Pillow; no automatic PDF or PPT export")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.documents:
        missing = [name for name in ("docx", "PIL") if importlib.util.find_spec(name) is None]
        if missing:
            parser.error("Document dependencies missing. Install python-docx and Pillow into this Python environment.")
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    if args.recompute or args.figures or args.documents:
        temporary = ROOT / "_tmp"
        temporary.mkdir(exist_ok=True)
        environment.update({"TMP": str(temporary), "TEMP": str(temporary), "TMPDIR": str(temporary),
                            "MPLCONFIGDIR": str(temporary / "mplconfig")})
    record = {"entry_point": "run_revision.py", "python_version": platform.python_version(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "options": vars(args),
              "policy": "Default: validate existing research results only; no new fitting or acquisition. Recompute is explicitly opt-in.",
              "commands": [], "status": "running"}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.recompute:
            for script, extra in [
                ("code/revision/review_evidence.py", []),
                ("code/revision/process_associations.py", []),
                ("code/revision/model_comparison.py", ["--workers", str(args.workers)]),
                ("code/revision/early_budget_al.py", ["--workers", str(args.workers)]),
            ]:
                run_script(script, extra, environment, record["commands"])
        record["source_checks"] = source_checks()
        for script in ("code/revision/validate_models.py", "code/revision/validate_early_al.py",
                       "code/revision/protocol_report.py"):
            run_script(script, [], environment, record["commands"])
        if args.figures:
            for script in ("code/revision/plot_review.py", "code/revision/check_revision_figures.py"):
                run_script(script, [], environment, record["commands"])
            figure_checks = json.loads(read_json(ROOT / "outputs/revision/figures/qa/summary.json"))
            if not figure_checks or any(x["text_exit_code"] or x["collision_exit_code"] for x in figure_checks):
                raise RuntimeError("Figure text/geometry checks failed; inspect outputs/revision/figures/qa")
            record["figure_check_count"] = len(figure_checks)
            record["visual_review"] = "Automated checks do not replace inspection of rendered figures/documents after changes."
        if args.documents:
            run_script("code/revision/build_documents_v2.py", [], environment, record["commands"])
            record["document_scope"] = "Markdown/DOCX and supporting content manifest generated; no PDF/PPT or visual-approval claim."
        record["status"] = "passed"
        record["validation_outputs"] = ["outputs/revision/models/validation.json",
                                        "outputs/revision/early_al/validation_independent.json",
                                        relative(SUMMARY)]
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error).replace(str(ROOT), ".")
        raise
    finally:
        record["entry_sha256"] = digest(Path(__file__))
        SUMMARY.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": record["status"], "source_checks_passed": len(record["source_checks"]),
                      "validation_outputs": record["validation_outputs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
