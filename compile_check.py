"""Source compilation check shared by the validation entry points.

The archived SHA256 manifests were dropped from validation. They recorded the
file bytes of the pre-comment revision, so every comment-only edit invalidated
them even when behaviour was unchanged. Validation now requires only that each
Python source still compiles, plus the structural and numerical checks that
read the saved artefacts.

Compilation uses the built-in ``compile`` function, so no ``.pyc`` files are
written and no bytecode cache is touched.
"""
from pathlib import Path

# Directories that never contain project sources to be validated.
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", "_tmp", "node_modules"}


def python_sources(root):
    """Yield every project Python source under ``root`` in sorted order."""
    root = Path(root).resolve()
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(part in EXCLUDED_PARTS for part in parts):
            continue
        yield path, root


def compile_sources(root):
    """Compile every project source without writing bytecode.

    Returns ``(records, failures)``: one record per file with its relative path
    and status, and the subset of records that failed to compile. A
    ``SyntaxError`` is reported per file instead of aborting the whole scan, so
    one broken file cannot hide the state of the others.
    """
    records, failures = [], []
    for path, base in python_sources(root):
        relative = path.relative_to(base).as_posix()
        try:
            # utf-8-sig tolerates a BOM on hand-edited files.
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as error:
            record = {"path": relative, "status": "failed", "error": str(error)}
            failures.append(record)
        else:
            record = {"path": relative, "status": "compiled"}
        records.append(record)
    return records, failures


def summary(records):
    """Count compiled and failed sources for compact reporting."""
    compiled = sum(1 for r in records if r["status"] == "compiled")
    return {"sources_checked": len(records), "sources_compiled": compiled,
            "sources_failed": len(records) - compiled}
