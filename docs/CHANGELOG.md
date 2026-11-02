# CHANGELOG.md

## [0.1.0] - 2026-11-02

### Added
- Project directory structure (src/, docs/, tests/, outputs/, legacy/, configs/)
- PROJECT_CONTEXT.md — persistent project context for future sessions
- REPOSITORY_AUDIT.md — complete code and data audit
- IMPLEMENTATION_PLAN.md — phased implementation roadmap
- DATA_DICTIONARY.md — feature definitions and data sources
- FEATURE_REGISTRY.md — feature groups and experiment assignments
- TASK_BOARD.md — active work tracking
- CHANGELOG.md — this file
- Preserved original code in legacy/ directory
- Python virtual environment with core dependencies

### Identified Issues
- 15+ code quality and methodology issues documented in REPOSITORY_AUDIT.md
- Missing UCSD review-level data (~1GB) blocks clustering pipeline
- Missing PPT file prevents claim verification
- Linear SVM instead of RBF SVM
- No stratification in train/test split
- No cross-validation
- Missing PR-AUC and precision metrics
- Missing Review+Network feature experiment

