"""Create a new byte-bound continuation source ledger; no native execution."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.meshprep.provenance import continuation_ledger

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    value=continuation_ledger(args.result)
    # Exclusive output; never overwrite a producer, input, or prior ledger.
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
