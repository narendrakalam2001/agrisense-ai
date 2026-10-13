"""Convenience launcher: `python scripts/run_api.py` starts the FastAPI server.

Equivalent to running:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
