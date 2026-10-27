"""Continue a recorded native preparation or render its inspected comparison."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.video_masks.region_comparison import continue_native, analyse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['continue','analyse'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--geometry-name',default='geometry02')
    args=parser.parse_args()
    if args.action=='continue':
        result=continue_native(args.run,args.geometry_name)
        return 0 if result.get('status')=='NATIVE_EXPERIMENTS_COMPLETE_NOT_ACCEPTED' else 1
    analyse(args.run,args.geometry_name)
    from integrations.video_masks.region_report import render
    render(args.run)
    return 0

if __name__=='__main__':raise SystemExit(main())
