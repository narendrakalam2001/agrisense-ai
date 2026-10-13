"""Convenience launcher: `python scripts/run_app.py` starts the Streamlit portal.

Equivalent to running:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "frontend" / "app.py")],
        check=True,
    )
