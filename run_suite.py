"""Run the declared main experiments and source/schedule sensitivities sequentially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = [
    ("real_main", "configs/real_main.json", []),
    ("cfd_main", "configs/cfd_main.json", []),
    ("real_drop2", "configs/real_main.json", ["--exclude-ids", "2"]),
    ("real_drop13", "configs/real_main.json", ["--exclude-ids", "13"]),
    ("real_drop24", "configs/real_main.json", ["--exclude-ids", "24"]),
    ("real_historical_schedule", "configs/real_historical_schedule.json", []),
    ("cfd_legacy_code", "configs/cfd_main.json", ["--cfd-variant", "legacy_code"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--start-at", choices=[r[0] for r in RUNS])
    args = parser.parse_args()
    (ROOT/"outputs/logs").mkdir(parents=True, exist_ok=True)
    records = []
    enabled = args.start_at is None
    expected_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT/"code/research").glob("*.py"))}
    expected_hashes["run_research.py"] = hashlib.sha256((ROOT/"run_research.py").read_bytes()).hexdigest()
    for name, config, extra in RUNS:
        if name == args.start_at:
            enabled = True
        path = ROOT/"outputs/results"/name/"run_metadata.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        expected_configuration = json.loads((ROOT/config).read_text(encoding="utf-8-sig"))
        expected_configuration["experiment"]["repeats"] = args.repeats
        expected_configuration["exclude_ids"] = (
            [int(value) for value in extra[extra.index("--exclude-ids")+1:]]
            if "--exclude-ids" in extra else [])
        expected_configuration["cfd_variant"] = (
            extra[extra.index("--cfd-variant")+1] if "--cfd-variant" in extra else "excel")
        source_file = ROOT/"data/raw"/(
            "user_confirmed_table_2026-11-09.json" if expected_configuration["dataset"] == "real"
            else "cfd_legacy_snapshot.json")
        complete = bool(previous and previous.get("status") == "complete"
                        and set(previous.get("stages_completed", [])) == {"basic", "al", "optimization"}
                        and previous["configuration"] == expected_configuration
                        and previous.get("code_sha256") == expected_hashes
                        and hashlib.sha256(source_file.read_bytes()).hexdigest() == previous["source_sha256"])
        if complete:
            print(f"REUSE VERIFIED COMPLETE {name}", flush=True)
            records.append({"run_name": name, "status": "complete_existing", "metadata": str(path),
                            "original_command": previous.get("command"),
                            "original_elapsed_seconds": previous.get("elapsed_seconds")})
            continue
        if not enabled:
            records.append({"run_name": name, "status": "not_requested_before_start"})
            continue
        cmd = [sys.executable, "-B", str(ROOT/"run_research.py"), "--config", config,
               "--run-name", name, "--workers", str(args.workers), "--repeats", str(args.repeats)] + extra
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        with (ROOT/"outputs/logs"/f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line}", end="", flush=True)
            code = process.wait()
        records.append({"run_name": name, "status": "complete" if code == 0 else "failed",
                        "exit_code": code, "elapsed_seconds": time.perf_counter()-started,
                        "command": cmd, "metadata": str(path)})
        (ROOT/"outputs/results/suite_manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        if code:
            raise SystemExit(code)
    (ROOT/"outputs/results/suite_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL REQUESTED RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
