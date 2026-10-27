"""Read-only photo-track applicability diagnostic; no native reconstruction."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-run', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    from integrations.demo_pipeline.parallel_photo_audit import audit
    try:
        result = audit(args.sfm_run, args.model_id, args.output)
        print(json.dumps(dict(status=result['status'], exact_status=result['exact_status'],
                              elapsed_seconds=time.monotonic()-start, output=str(args.output.resolve())), ensure_ascii=False))
        return 0  # diagnostic completed, not a model-acceptance exit code
    except Exception as exc:
        print(f'DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
