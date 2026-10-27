"""Existing native SfM -> fixed six-point research JSON/HTML; no reconstruction."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from integrations.demo_pipeline.research_code6 import export_native_code6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-run", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=10000)
    parser.add_argument("--no-controls", action="store_true", help="仅表示建码；默认另生成已知仿射正/负对照")
    parser.add_argument("--control-permutations", type=int, default=720, help="固定六点负对照最多1..720双射，预算未穷尽可UNKNOWN")
    args = parser.parse_args()
    try:
        result = export_native_code6(args.sfm_run, args.model_id, args.output, max_entries=args.max_entries,
                                     controls=not args.no_controls, max_control_permutations=args.control_permutations)
        print(json.dumps({key: result.get(key) for key in ("status", "native_point_ids", "counts", "controls", "wall_seconds", "error")}, ensure_ascii=False))
        return 0 if result["status"] == "SUCCESS" else 1
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
