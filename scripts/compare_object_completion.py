"""Automatically compare bound depth reconstruction and two inference policies."""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inferred-run',required=True);p.add_argument('--baseline-delivery');p.add_argument('--output',required=True);p.add_argument('--blender',action='store_true');p.add_argument('--seconds',type=int,default=600);a=p.parse_args(argv)
    from integrations.mvs_surface.strict_completion_comparison import run
    r=run(a.inferred_run,a.output,baseline_delivery=a.baseline_delivery,blender=a.blender,seconds=a.seconds);print(json.dumps(dict(status=r['status'],strict_closed=r['strict_closed'],wall_seconds=r['wall_seconds']),ensure_ascii=False));b=r['blender'];return 0 if not a.blender or b['status']=='SUCCESS' or b==dict(status='NOT_APPLICABLE',reason='strict_envelope_absent') else 1
if __name__=='__main__':raise SystemExit(main())
