"""
Script to create MAJOR_PROJECT_FINAL_STABLE.zip.
Includes:
- src/
- tests/
- docs/
- configs/
- outputs/models/
- outputs/experiments/
- outputs/figures/
- top-level markdown documentation and registries
- Dockerfile, docker-compose.yml, requirements.txt, .env.example
- small reference labeled product dataset (616 KB) for instant pipeline execution

Excludes:
- .git/
- venv/
- __pycache__/ and *.pyc
- .pytest_cache/
- huge raw gz files (> 5MB)
- other zip files
"""

import hashlib
import zipfile
from pathlib import Path

root_dir = Path(".").resolve()
zip_name = "MAJOR_PROJECT_FINAL_STABLE.zip"
zip_path = root_dir / zip_name

# Excluded patterns
EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "CHATGPT_AUDIT_DOCUMENTATION",
    "CHATGPT_PROJECT_AUDIT_BUNDLE",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".DS_Store",
    ".zip",
}

# Files larger than 5MB to exclude (raw text feats & raw UCSD archive)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

print(f"Building {zip_name} from {root_dir}...")

file_count = 0
total_uncompressed_bytes = 0

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(root_dir.rglob("*")):
        if path.is_dir():
            continue
        
        # Check excluded directories in path
        rel_parts = path.relative_to(root_dir).parts
        if any(ex in rel_parts for ex in EXCLUDE_DIRS):
            continue
        
        # Check excluded extensions
        if path.suffix in EXCLUDE_EXTENSIONS:
            continue
            
        # Check file size limit for raw dataset files
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

