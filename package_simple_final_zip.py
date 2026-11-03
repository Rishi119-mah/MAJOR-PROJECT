#!/usr/bin/env python3
"""
Script to package MAJOR_PROJECT_SIMPLE_FINAL.zip.

Includes:
- src/ (FastAPI backend, ML pipelines, graph models, evaluation, explainability)
- tests/ (Full backend pytest suite - 37 tests)
- frontend/ (Clean 4-page React 18 / TypeScript / Vite source code and tests)
- docs/ (Architecture, API contracts, model registries)
- configs/
- scripts/ (verify_model_registry.py, generate_test_report.py, capture_screenshots.py)
- outputs/models/ (All 21 registered serialized model pipelines)
- outputs/experiments/ (All empirical CSV benchmarks, cluster summaries, SHAP rankings)
- outputs/screenshots/ (Live screenshots of the 4 simplified application pages)
- data/ (Reference datasets)
- start_application.sh (Unified full-stack launcher)
- Root documentation (README.md, Dockerfile, requirements.txt, etc.)
"""

import hashlib
import zipfile
from pathlib import Path

root_dir = Path(".").resolve()
zip_name = "MAJOR_PROJECT_SIMPLE_FINAL.zip"
zip_path = root_dir / zip_name

EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "CHATGPT_AUDIT_DOCUMENTATION",
    "CHATGPT_PROJECT_AUDIT_BUNDLE",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".DS_Store",
    ".zip",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit for external raw files

print(f"Packaging {zip_name} from {root_dir}...")

file_count = 0
total_uncompressed_bytes = 0

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(root_dir.rglob("*")):
        if path.is_dir():
            continue

        rel_parts = path.relative_to(root_dir).parts

        # Check directory exclusions
        if any(ex in rel_parts for ex in EXCLUDE_DIRS):
            continue

        # Check file extension exclusions
        if path.suffix in EXCLUDE_EXTENSIONS:
            continue

        file_size = path.stat().st_size
        # Check file size limit for raw datasets (skips > 5MB raw files, keeps all model pipelines)
        if path.suffix == ".gz" and file_size > MAX_FILE_SIZE:
            print(f"  [SKIPPED OVERSIZED DATASET] {path.relative_to(root_dir)} ({file_size / (1024*1024):.2f} MB)")
            continue

        arcname = str(path.relative_to(root_dir))
        zf.write(path, arcname)
        file_count += 1
        total_uncompressed_bytes += file_size

zip_size_bytes = zip_path.stat().st_size
sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()

print("\n" + "=" * 60)
print(f"PACKAGING COMPLETE: {zip_name}")
print(f"  - Total files included: {file_count}")
print(f"  - Total uncompressed size: {total_uncompressed_bytes / (1024*1024):.2f} MB")
print(f"  - Final ZIP archive size: {zip_size_bytes / (1024*1024):.2f} MB")
print(f"  - SHA-256 Checksum: {sha256}")
print("=" * 60)
