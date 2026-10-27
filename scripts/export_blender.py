"""Export validated PLY scenes as a native Blender file."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.blender_export import export_blend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender-exe", type=Path)
    parser.add_argument("--profile", choices=("standard", "surface400k"), default="standard")
    args = parser.parse_args()
    report = export_blend(args.spec, args.output, blender_exe=args.blender_exe, profile=args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
