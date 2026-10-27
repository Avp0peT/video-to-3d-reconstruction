"""Review every selected view without running a model or rewriting its source."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.silhouette_review import review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run')
    source.add_argument('--experimental-surface')
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=int, default=180)
    args = p.parse_args()
    try:
        if args.experimental_surface:
            from integrations.demo_pipeline.experimental_silhouette import review_experimental
            result = review_experimental(args.experimental_surface, args.output, seconds=args.seconds)
        else:
            result = review(args.run, args.output, seconds=args.seconds)
    except (ValueError, OSError, KeyError, TypeError, TimeoutError) as exc:
        print(json.dumps({'status': 'FAILED', 'error': type(exc).__name__ + ': ' + str(exc)}))
        return 2
    print(json.dumps({k: result[k] for k in ('status', 'completed_views', 'expected_views', 'IoU_min', 'IoU_median', 'IoU_max', 'elapsed_seconds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
