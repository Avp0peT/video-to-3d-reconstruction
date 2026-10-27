"""Prepare generic motion-supported mask windows; no neural or native 3D run."""
import os
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from integrations.video_masks.motion_part_pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window-size', type=int, default=6)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--raw-motion-only', dest='coherence', action='store_false',
                        help='Explicit control: skip connected temporal coherence refinement')
    args = parser.parse_args()
    try:
        result = run(**vars(args))
    except Exception as exc:
        print(json.dumps(dict(status='FAILED',error=type(exc).__name__+': '+str(exc)), ensure_ascii=False))
        return 2
    print(json.dumps({k: result[k] for k in ('status','frame_count','window_count','elapsed_seconds')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
