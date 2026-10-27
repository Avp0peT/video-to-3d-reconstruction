"""Automatically audit whole/local geometry and export an additive supported model."""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from integrations.mvs_surface.mutual_model_pipeline import run
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-model',required=True);p.add_argument('--output',required=True);p.add_argument('--blender',action='store_true');p.add_argument('--local-policy',choices=('supported-only','all-points'),default='supported-only');p.add_argument('--global-policy',choices=('supported-only','all-points'),default='supported-only');p.add_argument('--face-policy',choices=('supported-only','original'),default='supported-only');a=p.parse_args(argv)
    if a.face_policy=='original' and a.local_policy!='all-points':p.error('--face-policy original requires --local-policy all-points')
    r=run(a.source_model,a.output,blender=a.blender,local_policy=a.local_policy,global_policy=a.global_policy,face_policy=a.face_policy)
    print(json.dumps({k:r[k] for k in ('status','global_counts','local_counts','fusion_counts','wall_seconds')},ensure_ascii=False))
    return 0 if r['status'] in ('SUPPORTED_MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED','ALL_LOCAL_COMPARISON_NOT_GEOMETRY_ACCEPTED','ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED','RELAXED_ADMISSION_COMPARISON_NOT_GEOMETRY_ACCEPTED') and (not a.blender or r['blender']['status']=='SUCCESS') else 1
if __name__=='__main__':raise SystemExit(main())
