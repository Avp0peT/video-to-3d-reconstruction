"""Read-only native camera domain gate, run under the pipeline supervisor."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.camera_domain_gate import audit_camera_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('audit output already exists')
    result = audit_camera_domain(args.model)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return {'CAMERA_DOMAIN_VALID': 0, 'CAMERA_DOMAIN_REJECTED': 1}.get(result['status'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
