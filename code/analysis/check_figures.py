"""Run source, PDF text and collision checks; cache only unchanged final PDFs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
os.environ["PYTHONUTF8"]="1"
os.environ["PYTHONIOENCODING"]="utf-8"
TOOLS=ROOT/"code/tools"
QA=ROOT/"outputs/reports/figure_qa"
QA.mkdir(parents=True,exist_ok=True)
source=subprocess.run([sys.executable,str(TOOLS/"validate_figure.py"),str(ROOT/"code/analysis/plot_research.py"),"--json"],capture_output=True,text=True,encoding="utf-8")
(QA/"source_validation.json").write_text(source.stdout,encoding="utf-8")
records=[]
for pdf in sorted((ROOT/"outputs/figures").glob("*.pdf")):
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    result_path=QA/f"{pdf.stem}.checks.json"
    if result_path.exists():
        old=json.loads(result_path.read_text(encoding="utf-8"))
        if old["sha256"]==sha:
            records.append(old)
            continue
    text_check=subprocess.run([sys.executable,str(TOOLS/"audit_pdf_text.py"),str(pdf),"--min-pt","5","--json"],capture_output=True,text=True,encoding="utf-8")
    (QA/f"{pdf.stem}.text.json").write_text(text_check.stdout,encoding="utf-8")
    collision=subprocess.run([sys.executable,str(TOOLS/"audit_figure_collisions.py"),str(pdf),"--json-out",str(QA/f"{pdf.stem}.collision.json"),"--overlay-pdf",str(QA/f"{pdf.stem}.collision.pdf")],capture_output=True,text=True,encoding="utf-8")
    (QA/f"{pdf.stem}.collision.log").write_text(collision.stdout+collision.stderr,encoding="utf-8")
    record={"figure":str(pdf.relative_to(ROOT)),"sha256":sha,"text_exit_code":text_check.returncode,"collision_exit_code":collision.returncode}
    result_path.write_text(json.dumps(record,indent=2),encoding="utf-8")
    records.append(record)
(QA/"summary.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print(json.dumps({"pdf_count":len(records),"nonzero_checks":[r for r in records if r["text_exit_code"] or r["collision_exit_code"]]},ensure_ascii=False))
