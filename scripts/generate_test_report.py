#!/usr/bin/env python3
"""
Generate Verified Pytest Test Report Artifact.

Runs pytest programmatically across tests/, collects individual test outcomes,
and exports a structured JSON artifact to outputs/test_report.json.
This ensures frontend reports and academic sign-offs display verified test telemetry.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


class TestReportCollector:
    """Pytest plugin hook collector for test run metrics."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.total = 0
        self.test_records = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1

            self.test_records.append({
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration": round(report.duration, 4),
            })
        elif report.when == "setup" and report.skipped:
            self.total += 1
            self.skipped += 1
            self.test_records.append({
                "nodeid": report.nodeid,
                "outcome": "skipped",
                "duration": 0.0,
            })


def generate_report():
    collector = TestReportCollector()
    start_time = time.time()

    # Prevent matplotlib permission issues
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"

    exit_code = pytest.main(["tests", "-v"], plugins=[collector])
    elapsed = time.time() - start_time

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "test_report.json"

    report_payload = {
        "status": "passed" if exit_code == 0 else "failed",
        "total_tests": collector.total,
        "passed": collector.passed,
        "failed": collector.failed,
        "skipped": collector.skipped,
        "duration_seconds": round(elapsed, 2),
        "test_execution_date": datetime.now(timezone.utc).isoformat(),
        "environment": f"Python {sys.version.split()[0]} • pytest {pytest.__version__} • {platform.system()}",
        "test_command": "pytest tests/ -v",
        "tests": collector.test_records,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[Test Report] Generated verified artifact at {report_path}")
    print(f"Summary: {collector.passed}/{collector.total} passed ({collector.failed} failed, {collector.skipped} skipped) in {elapsed:.2f}s")
    return exit_code


if __name__ == "__main__":
    code = generate_report()
    sys.exit(code)

