"""Export the dominant closed inferred component with preserved source layers."""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inferred-run',required=True);p.add_argument('--output',required=True);p.add_argument('--blender',action='store_true');p.add_argument('--seconds',type=int,default=600);a=p.parse_args(argv)
    from integrations.mvs_surface.completion_delivery import finalize_completion
    r=finalize_completion(a.inferred_run,a.output,blender=a.blender,seconds=a.seconds);print(json.dumps(dict(status=r['status'],wall_seconds=r['wall_seconds']),ensure_ascii=False));return 0 if r['closed_estimated_outer_model'] else 1
if __name__=='__main__':raise SystemExit(main())
