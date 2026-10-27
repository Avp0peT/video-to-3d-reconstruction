"""Run the explicitly instrumented SAM3 profile under an outer Job supervisor."""
from pathlib import Path
import argparse
import os
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.runtime import execute_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return execute_config(args.config, score_diagnostics=True)


if __name__ == "__main__":
    raise SystemExit(main())
