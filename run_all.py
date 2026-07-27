"""One-command reproducibility entry point for numerical results and figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "src" / script)]
    print(f"\nRunning: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    run("run_experiments.py")
    run("make_figures.py")
    print("\nCompleted. See results/ for tables, figures, and the experiment summary.")

