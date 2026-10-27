"""Review an existing raw depth-fusion result and export nonregressing additions."""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from integrations.mvs_surface.depth_fusion_finalize import finalize
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fusion-run',required=True);p.add_argument('--output',required=True);p.add_argument('--blender',action='store_true');a=p.parse_args(argv)
    r=finalize(a.fusion_run,a.output,blender=a.blender)
    print(json.dumps(dict(status=r['status'],admission=r['admission'],wall_seconds=r['wall_seconds']),ensure_ascii=False))
    return 0 if not a.blender or r['blender']['status']=='SUCCESS' else 1
if __name__=='__main__':raise SystemExit(main())
