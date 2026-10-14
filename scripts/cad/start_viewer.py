"""Launch the project Viewer through run_memory_limited.py.

Usage: .venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/start_viewer.py
The exec preserves the writable cache, single worker settings and address-space
limit, while releasing the launcher's non-inherited lock for later CAD tasks.
"""
import os
from pathlib import Path
import resource
import sys

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit("Start through scripts/cad/run_memory_limited.py")
os.environ["CADGEN_JOBS"] = "1"
os.chdir(Path(__file__).resolve().parents[2] / "cad/local_fix")
os.execv(sys.executable, [sys.executable, "-m", "cadgen.viewer",
                        "--host", "0.0.0.0", "--json", *sys.argv[1:]])
