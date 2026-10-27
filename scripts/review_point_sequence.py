"""Automatically review current-run temporal points, masks and saved-camera geometry."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from integrations.demo_pipeline.automatic_point_review import review
    try:
        result = review(args.pipeline, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'INPUT_ERROR: {exc}', file=sys.stderr); return 2
    print(json.dumps({k:result[k] for k in ('status','reason','result_path','report_path')}, ensure_ascii=False))
    return {'AUTO_POINT_REVIEW_COMPLETE':0, 'INSUFFICIENT_DATA':1, 'RESOURCE_LIMIT':5}.get(result['status'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
