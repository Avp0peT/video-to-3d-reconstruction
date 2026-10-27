"""Photo object entrypoint: fix native runtime threads before importing NumPy.

Use the project's pinned reconstruction interpreter:
python -B scripts/reconstruct_object.py --io-run <photo-io-v1-run> --output <new-output-parent>
This sparse stage does not claim to produce a CAD solid.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kozlov_lab.reconstruction.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(arg == "--profile" or arg.startswith("--profile=") for arg in arguments):
        arguments += ["--profile", "cad-object-v1"]
    raise SystemExit(main(arguments))
