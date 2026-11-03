#!/usr/bin/env python3
"""
Script to package MAJOR_PROJECT_FINAL_RELEASE.zip.

Includes:
- src/ (FastAPI backend, ML pipelines, graph models, evaluation, explainability)
- tests/ (Full backend pytest suite - 37 tests)
- frontend/ (Complete React 18 / TypeScript / Vite source code, tests, and dist/)
- docs/ (Architecture, API contracts, model registries)
- configs/
- scripts/ (verify_model_registry.py, generate_test_report.py, capture_screenshots.py)
- outputs/models/ (All 21 registered serialized model pipelines)
- outputs/experiments/ (All empirical CSV benchmarks, cluster summaries, SHAP rankings)
- outputs/figures/ (SHAP summary visualizations)
- outputs/screenshots/ (Full-page live screenshots of all 9 application pages)
- outputs/test_report.json (Verified pytest run report)
- data/ (Reference datasets)
- start_application.sh (Unified full-stack launcher)
- Root documentation and registries (README.md, HARD_CODED_DATA_AUDIT.md, FINAL_TEST_REPORT.md, etc.)
"""

import hashlib
import zipfile
from pathlib import Path

root_dir = Path(".").resolve()
zip_name = "MAJOR_PROJECT_FINAL_RELEASE.zip"
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
    "CHATGPT_AUDIT_DOCUMENTATION",
    "CHATGPT_PROJECT_AUDIT_BUNDLE",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".DS_Store",
    ".zip",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit for external raw datasets (> 5MB)

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

        # Check file size limit for raw datasets (skips > 5MB UCSD raw files)
        if path.suffix == ".gz" and path.stat().st_size > MAX_FILE_SIZE:
            print(f"Skipping large dataset: {path.name} ({path.stat().st_size / (1024*1024):.1f} MB)")
            continue

        rel_path = path.relative_to(root_dir)
        zf.write(path, arcname=str(rel_path))
        file_count += 1
        total_uncompressed_bytes += path.stat().st_size

# Compute SHA-256
sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        sha256.update(chunk)

zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
uncompressed_mb = total_uncompressed_bytes / (1024 * 1024)
checksum = sha256.hexdigest()

print(f"\nSuccessfully built {zip_name}:")
print(f"  Files Packed       : {file_count}")
print(f"  Uncompressed Size  : {uncompressed_mb:.2f} MB")
print(f"  Compressed Size    : {zip_size_mb:.2f} MB")
print(f"  SHA-256 Checksum   : {checksum}")

