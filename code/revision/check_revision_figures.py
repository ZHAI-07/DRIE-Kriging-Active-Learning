"""Run PDF geometry/text checks for revision figures only."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ["PYTHONUTF8"]="1"
os.environ["PYTHONIOENCODING"]="utf-8"
TOOLS=ROOT/"code/tools"
QA=ROOT/"outputs/revision/figures/qa"
QA.mkdir(parents=True,exist_ok=True)
source=subprocess.run([sys.executable,str(TOOLS/"validate_figure.py"),str(ROOT/"code/revision/plot_review.py"),"--json"],capture_output=True,text=True,encoding="utf-8")
(QA/"source_validation.json").write_text(source.stdout,encoding="utf-8")
records=[]
for pdf in sorted((ROOT/"outputs/revision/figures").glob("*.pdf")):
    text=subprocess.run([sys.executable,str(TOOLS/"audit_pdf_text.py"),str(pdf),"--min-pt","5","--json"],capture_output=True,text=True,encoding="utf-8")
    (QA/f"{pdf.stem}.text.json").write_text(text.stdout,encoding="utf-8")
    collision=subprocess.run([sys.executable,str(TOOLS/"audit_figure_collisions.py"),str(pdf),"--json-out",str(QA/f"{pdf.stem}.collision.json")],capture_output=True,text=True,encoding="utf-8")
    (QA/f"{pdf.stem}.collision.log").write_text(collision.stdout+collision.stderr,encoding="utf-8")
    records.append({"figure":str(pdf.relative_to(ROOT)),"sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    "text_exit_code":text.returncode,"collision_exit_code":collision.returncode})
(QA/"summary.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print(json.dumps({"figures":len(records),"failed":[x for x in records if x["text_exit_code"] or x["collision_exit_code"]]}))
