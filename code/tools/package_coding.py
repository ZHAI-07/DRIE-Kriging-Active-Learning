"""Create the scientific-code ZIP, including the data needed for default checks."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'DRIE_Coding_代码包_2026-11-10.zip'
PREFIX='DRIE_Coding'
sha=lambda b:hashlib.sha256(b).hexdigest()


def select_files():
    files={}
    def add(rel,arc=None):
        p=ROOT/rel
        if not p.is_file():raise FileNotFoundError(rel)
        files[arc or rel]=p
    add('README_CODING.md','README.md')
    for rel in ['README_CODING.md','AGENTS.md','requirements_coding.txt','requirements.lock.txt',
                'run_coding.py','run_submission.py','run_revision.py','run_research.py',
                'run_suite.py','summarize_runs.py','verify_outputs.py','code/tools/package_coding.py',
                'data/raw/user_confirmed_table_2026-11-09.json','data/raw/cfd_legacy_snapshot.json',
                'data/metadata/user_clarifications.json','outputs/reports/duw_recovery.json']:
        add(rel)
    for folder,suffix in [('code/research','.py'),('tests','.py'),('configs','.json')]:
        for p in (ROOT/folder).glob('*'+suffix):add(p.relative_to(ROOT).as_posix())
    for name in ['model_comparison.py','early_budget_al.py','process_associations.py',
                 'review_evidence.py','review_v3.py','validate_models.py','validate_early_al.py',
                 'plot_review.py','check_revision_figures.py']:
        add('code/revision/'+name)
    for name in ['plot_research.py','check_figures.py']:add('code/analysis/'+name)
    for name in ['audit_panel_alignment.py','audit_pdf_text.py','audit_figure_collisions.py','validate_figure.py']:
        add('code/tools/'+name)
    for folder in ['outputs/revision/models','outputs/revision/early_al','outputs/revision/evidence',
                   'outputs/revision_v3/analysis','outputs/coding/validation']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.csv','.json'} and p.name!='package_validation.json':
                add(p.relative_to(ROOT).as_posix())
    for run in ['real_main','real_drop2','real_drop13','real_drop24','real_historical_schedule','cfd_main','cfd_legacy_code']:
        for p in (ROOT/'outputs/results'/run).iterdir():
            if p.is_file() and p.suffix in {'.csv','.json'}:add(p.relative_to(ROOT).as_posix())
    add('outputs/revision_v4/validation/full_reproduction.json')
    return files


def main():
    payload={name:p.read_bytes() for name,p in select_files().items()}
    assert all(Path(name).suffix not in {'.docx','.pptx','.pdf','.png','.svg','.zip'} for name in payload)
    assert not any('/__pycache__/' in p or '/.venv/' in p or p.startswith('_tmp/') for p in payload)
    manifest={'scope':'Scientific code, README, dependencies, configurations, tests and required numerical evidence',
              'files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(payload.items())]}
    payload['PACKAGE_MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(payload.items()):z.writestr(PREFIX+'/'+name,data)
    assert DEST.stat().st_size<30_000_000
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(payload)
        for item in manifest['files']:assert sha(z.read(PREFIX+'/'+item['path']))==item['sha256']
    record={'status':'passed','archive':DEST.name,'bytes':DEST.stat().st_size,'decimal_MB':DEST.stat().st_size/1e6,
            'files':len(payload),'sha256':sha(DEST.read_bytes()),'CRC_and_manifest':'passed','reports_and_slides_included':False}
    qa=ROOT/'outputs/coding/validation';qa.mkdir(parents=True,exist_ok=True)
    (qa/'package_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
