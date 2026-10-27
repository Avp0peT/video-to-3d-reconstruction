"""Explicit plan/run CLI for a bounded CPU OpenMVS surface recipe."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from integrations.mvs_surface.runner import make_plan, execute_plan, save_json, PROFILES
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    prepare = subs.add_parser('plan', help='validate inputs and save recipe; starts no native process')
    for name in ('scene-path', 'image-root', 'mask-root', 'validity-root', 'tool-root'):
        prepare.add_argument('--' + name, required=True)
    for name in ('total-seconds', 'stage-seconds', 'memory-bytes', 'output-bytes'):
        prepare.add_argument('--' + name, required=True, type=int)
    prepare.add_argument('--input-origin', choices=['unverified', 'synthetic', 'real', 'experimental-real'], default='unverified')
    prepare.add_argument('--ignore-label', type=int, choices=[0, -1], default=0)
    prepare.add_argument('--photo-source', help='explicit source specification; required for real or experimental-real')
    prepare.add_argument('--profile', choices=PROFILES, default=argparse.SUPPRESS)
    prepare.add_argument('--plan-output', required=True)
    run = subs.add_parser('run', help='execute frozen recipe once in a new output directory')
    run.add_argument('--plan', required=True); run.add_argument('--plan-sha256', required=True)
    run.add_argument('--output', required=True)
    args = vars(parser.parse_args()); action = args.pop('action')
    try:
        if action == 'plan':
            output = Path(args.pop('plan_output')).resolve()
            if output.exists():
                raise ValueError('plan output already exists')
            plan = make_plan(**args)
            save_json(output, plan, exclusive=True)
            print(json.dumps({'status': 'PLAN_ONLY', 'native_launches': 0, 'path': str(output),
                              'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
            return 0
        path = Path(args['plan']).resolve(strict=True)
        with path.open('rb') as stream:
            data = stream.read(4 * 1024**2 + 1)
        if len(data) > 4 * 1024**2 or hashlib.sha256(data).hexdigest() != args['plan_sha256']:
            raise ValueError('plan size/hash mismatch')
        def bad_constant(value):
            raise ValueError('nonfinite JSON constant: ' + value)
        plan = json.loads(data, parse_constant=bad_constant)
        if not isinstance(plan, dict):
            raise ValueError('plan must be an object')
        result = execute_plan(plan, args['output'])
        print(json.dumps({'status': result['status'], 'exit_code': result['exit_code'],
                          'native_launches': result['native_launches']}))
        return result['exit_code']
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'INPUT_ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
