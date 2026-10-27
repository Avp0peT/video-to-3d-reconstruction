"""Review sampled-video LK candidate tracks without rerunning reconstruction."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True,type=Path,help='existing committed video pipeline directory')
    parser.add_argument('--output',required=True,type=Path,help='fresh review directory outside protected inputs')
    args=parser.parse_args(argv)
    try:
        from integrations.demo_pipeline.post_review import check_output_scope,read
        pipeline=args.pipeline.resolve(strict=True);output=args.output.resolve()
        if not pipeline.is_dir() or output.exists() or output==pipeline or output in pipeline.parents:
            raise ValueError('existing pipeline and fresh disjoint output required')
        discovery=read(pipeline/'pipeline.json')
        check_output_scope(pipeline,output,discovery)
        from integrations.demo_pipeline.temporal_review import review_video
        result=review_video(pipeline,output,discovery=discovery)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'INPUT_ERROR: {exc}',file=sys.stderr)
        return 2
    print(json.dumps({key:result.get(key) for key in ('status','reason','result_path','report_path')},
        ensure_ascii=False,allow_nan=False))
    return {'DIAGNOSTIC_COMPLETE':0,'NOT_AVAILABLE':1,'FAILED':2}.get(result.get('status'),2)


if __name__=='__main__':raise SystemExit(main())
