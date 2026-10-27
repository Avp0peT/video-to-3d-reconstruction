"""Explicit bounded CAD continuation of an existing unified photo result."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.cad import plan,execute

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--freecad-python',type=Path);p.add_argument('--python',type=Path)
    p.add_argument('--geometry-record',type=Path,help='immutable CAD_SOURCE.json emitted by unified target cad')
    p.add_argument('--plan-only',action='store_true',help='read-only preflight; no output directory or native launch')
    args=vars(p.parse_args(argv));only=args.pop('plan_only')
    try:
        frozen=plan(**args)
        if only:print(json.dumps(frozen,ensure_ascii=False,indent=2));return 0
        result=execute(frozen);print(json.dumps(result,ensure_ascii=False,indent=2));return result['exit_code']
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'INPUT_ERROR: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
