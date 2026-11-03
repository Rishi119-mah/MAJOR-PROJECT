#!/usr/bin/env python3
"""
Capture end-to-end full-page screenshots of the 4 simplified application pages.
Runs headless Chrome with virtual-time-budget to ensure full React hydration and chart rendering.
"""

import subprocess
import time
from pathlib import Path

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUTPUT_DIR = Path("outputs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("01_dashboard.png", "http://127.0.0.1:5173/dashboard"),
    ("02_product_checker.png", "http://127.0.0.1:5173/product-checker"),
    ("03_model_analysis.png", "http://127.0.0.1:5173/model-analysis"),
    ("04_network_analysis.png", "http://127.0.0.1:5173/network-analysis"),
]

def main():
    print("=" * 60)
    print("CAPTURING APPLICATION SCREENSHOTS FOR 4 SIMPLIFIED PAGES")
    print("=" * 60)

    for filename, url in PAGES:
        out_path = OUTPUT_DIR / filename
        print(f"Capturing: {url} -> {out_path} ...", end=" ", flush=True)

        cmd = [
            CHROME_BIN,
            "--headless",
            "--disable-gpu",
            "--window-size=1440,960",
            "--virtual-time-budget=3000",
            f"--screenshot={out_path.resolve()}",
            url,
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if out_path.exists() and out_path.stat().st_size > 5000:
                print(f"OK ({out_path.stat().st_size:,} bytes)")
            else:
                print(f"FAILED (Empty or missing: {res.stderr[:200]})")
        except Exception as e:
            print(f"ERROR: {e}")

    print("=" * 60)
    print("All screenshots captured:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
