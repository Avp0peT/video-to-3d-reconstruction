"""Scale existing verified photo CAD with a supplied real measurement record."""
import sys,argparse,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.measurement import plan,execute

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','reference','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--cad-output',type=Path);parser.add_argument('--freecad-python',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=vars(parser.parse_args(argv));only=args.pop('plan_only')
    try:
        frozen=plan(**args)
        if only:print(json.dumps(frozen,ensure_ascii=False,indent=2));return 0
        result=execute(frozen);print(json.dumps(result,ensure_ascii=False,indent=2));return result['exit_code']
    except (OSError,ValueError,KeyError,TypeError) as exc:print('INPUT_ERROR: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
