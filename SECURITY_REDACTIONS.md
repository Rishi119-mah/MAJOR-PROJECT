# Security and Credential Audit Report

## 1. Secret Scanning Audit
An automated regular-expression and keyword scan was conducted across the entire repository for:
- API keys (OpenAI, AWS, Google Cloud, Hugging Face)
- Access tokens and bearer tokens
- Passwords and private keys (`.pem`, `.key`)
- Database credentials and connection strings
- Personal access tokens (GitHub PATs)
- Sensitive personal data or internal credentials

## 2. Findings
- **Total Secrets Detected**: **0**
- **Total Files Redacted**: **0**
- **Scan Notes**:
  - Keyword search for `token` matched only NLP TF-IDF n-gram tokens (e.g. `'bluetooth'`, `'charg'`) in `docs/DATA_DICTIONARY.md` and feature columns in `data/product_level_data_text_feats.csv.gz`.
  - Binary strings inside CatBoost compiled pipelines matched standard serialization headers.
  - No secret tokens, API keys, or private environment files exist in this codebase.

## 3. Exclusion of Sensitive & Ephemeral Files
The following non-reproducible, temporary, or virtual environment files were strictly excluded from the audit archive:
- `venv/` and `.venv/` (Python virtual environments)
- `__pycache__/` and `*.pyc` (Python bytecode)
- `.git/` (Internal git revision history)
- `.pytest_cache/` (Local test runner cache)
- `catboost_info/` (CatBoost training logs)
- Large raw datasets (> 100 MB)
