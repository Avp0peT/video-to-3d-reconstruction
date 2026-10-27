"""Make a standalone, offline 3D preview from bounded, hash-bound PLY scenes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
from integrations.viewer.export import export_html


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new directory under project outputs")
    p.add_argument("--profile", choices=("standard", "surface400k"), default="standard",
                   help="explicit bounded display profile; no sampling or geometry changes")
    args = p.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or (ROOT/"outputs").resolve() not in output.parents:
        print("output must be new and strictly under project outputs", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        result = export_html(args.spec, output/"viewer.html", profile=args.profile)
        result["exit_code"] = 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        result = {"status": "REJECTED", "exit_code": 2, "error": f"{type(error).__name__}: {error}"}
    except Exception as error:
        result = {"status": "INTERNAL_ERROR", "exit_code": 4, "error": f"{type(error).__name__}: {error}"}
    result["wall_seconds"] = time.monotonic()-start
    (output/"RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(output/"RESULT.json")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
