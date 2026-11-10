"""Validate saved v2 research results by default; recomputation is opt-in.

Examples:
  python run_revision.py
  python run_revision.py --recompute --workers 4
  python run_revision.py --figures --documents

Default execution performs numerical/source checks and refreshes validation
records only. It does not retrain models or create new sampling experiments.
--documents builds portable Markdown/DOCX with standard python-docx/Pillow;
it never invokes Word, PDF export, Node, or the Codex presentation runtime.
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
SUMMARY = ROOT / "outputs/revision/reproduction_validation.json"


def relative(value):
    """Portable display of project-owned paths; never list machine drive paths."""
    return Path(value).relative_to(ROOT).as_posix()


def read_json(path):
    return path.read_text(encoding="utf-8-sig")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_file(recorded):
    path = (ROOT / str(recorded).replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("A recorded source path is outside the project")
    return path


def source_checks():
    """Check the new scientific source/implementation manifests, not layout QA."""
    checks = []
    def check(path, expected, manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source: {relative(path)}")
        if digest(path) != expected:
            raise RuntimeError(f"Changed source: {relative(path)} (declared by {manifest})")
        checks.append({"manifest": manifest, "path": relative(path), "status": "matched"})
    evidence = [
        ("outputs/revision/evidence/review_evidence_summary.json", "source_sha256", "code/revision/review_evidence.py"),
        ("outputs/revision/evidence/process_associations_metadata.json", "input_sha256", "code/revision/process_associations.py"),
    ]
    for name, key, script in evidence:
        meta = json.loads(read_json(ROOT / name))
        for source, expected in meta[key].items():
            check(project_file(source), expected, name)
        check(ROOT / script, meta["script_sha256"], name)
    name = "outputs/revision/early_al/protocol.json"
    meta = json.loads(read_json(ROOT / name))
    if meta.get("status") != "complete":
        raise RuntimeError("Early-budget AL execution is incomplete")
    for source, expected in meta["source_sha256"].items():
        check(project_file(source), expected, name)
    model_paths = sorted((ROOT / "outputs/revision/models").glob("*/run_metadata.json"))
    if len(model_paths) != 6:
        raise RuntimeError("Expected six completed v2 model scenarios")
    for path in model_paths:
        meta = json.loads(read_json(path))
        if meta.get("status") != "complete":
            raise RuntimeError(f"Incomplete scenario: {relative(path)}")
        check(project_file(meta["source"]), meta["source_sha256"], relative(path))
        check(ROOT / "code/revision/model_comparison.py", meta["code_sha256"], relative(path))
    return checks


def run_script(script, arguments, environment, commands):
    if not (ROOT / script).is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    display = [Path(sys.executable).name, "-B", script, *arguments]
    print("RUN " + " ".join(display), flush=True)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-B", script, *arguments], cwd=ROOT,
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    # Child scripts retain their own diagnostics; this entry keeps console
    # progress concise and makes copied commands independent of this host.
    output = result.stdout.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
    commands.append({"command": display, "exit_code": result.returncode,
                     "elapsed_seconds": round(time.perf_counter()-started, 3),
                     "output_tail": output[-1800:]})
    if result.returncode:
        print(output, file=sys.stderr)
        raise RuntimeError(f"Command failed: {script}")
    print("PASS " + script, flush=True)


def main():
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
        for script in ("code/revision/validate_models.py", "code/revision/validate_early_al.py"):
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
    print(json.dumps({"status": record["status"], "source_hashes_checked": len(record["source_checks"]),
                      "validation_outputs": record["validation_outputs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
