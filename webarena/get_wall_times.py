"""
Iterate through every `*.trace.zip` archive in the `tracedata/` directory,
extract it, run `get_trace.py` (which expects a `trace.trace` file in the
current working directory), grab the reported human wall-time, and collect
all wall-times in a list.  Temporary extraction directories are used so the
extra files (`trace.network`, `trace.stacks`, the huge `resources/` folder,
etc.) are deleted automatically when we're done.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TRACEDATA_DIR = ROOT_DIR / "tracedata"
TRACE_SCRIPT = ROOT_DIR / "get_trace.py"

WALLTIMES: list[float] = []

index = 0

for zip_path in sorted(TRACEDATA_DIR.glob("*.trace.zip")):
    index += 1
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Extract the entire archive (quick + keeps life simple)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)

        # 2. Run get_trace.py inside the temp dir
        result = subprocess.run(
            [sys.executable, str(TRACE_SCRIPT)],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # 3. Parse the printed wall-time (e.g. "Human time for this task: 123.4 seconds")
        match = re.search(
            r"Human time for this task:\s*([\d.]+)\s*seconds", result.stdout
        )
        if not match:
            print(f"Could not find wall-time in output of {zip_path.name}")
            continue

        WALLTIMES.append(float(match.group(1)))

    print(f"Wall-time for {index} {zip_path.name}: {WALLTIMES[-1]} seconds")

# 4. Done
print("Wall-times (seconds):")
print(sorted(WALLTIMES))
print(f"total indices: {index}")
