"""Select object-mask candidates and derive conditional local 3D from saved poses."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'

CHILD="""import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from integrations.demo_pipeline.coverage_expansion import run
r=run(*sys.argv[1:5], tracking_profile=sys.argv[5] or None, sfm_run=sys.argv[6] or None)
print(r['status'])
raise SystemExit(0 if r['status']=='PARTIAL_CONDITIONAL_EXPANSION' else 2)
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True,type=Path)
    parser.add_argument('--local-models',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--selection-scope',choices=('object','whole-frame'),default='object',
        help='object (default): seed/track only in committed masks; whole-frame is an explicit diagnostic mode')
    parser.add_argument('--tracking-profile',choices=('whole-video-incremental-v1',))
    parser.add_argument('--sfm-run',type=Path,help='explicit independently bound SfM run directory or run.json')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--blender',action='store_true')
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.partial_reconstruction import save
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    pipeline=args.pipeline.resolve(strict=True);local=args.local_models.resolve(strict=True)
    sfm_run=args.sfm_run.resolve(strict=True) if args.sfm_run else None
    if sfm_run and sfm_run.is_dir():sfm_run=sfm_run/'run.json'
    if sfm_run and not args.tracking_profile:raise ValueError('--sfm-run requires --tracking-profile')
    if args.tracking_profile and args.selection_scope!='object':raise ValueError('incremental profile requires object selection')
    output=args.output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (pipeline,local)):
        raise ValueError('fresh nonoverlapping output required')
    paths=[Path(__file__),ROOT/'src/kozlov_lab/stereo/coverage_tracks.py']
    paths += [ROOT/'integrations/demo_pipeline'/n for n in
        ('coverage_expansion.py','coverage_triangulation.py','partial_point_gate.py',
         'partial_reconstruction.py','partial_surface.py','temporal_review.py',
         'parallel_photo_audit.py','registration_probe.py','pipeline.py','supervision.py')]
    paths += [ROOT/'integrations/viewer'/n for n in ('export.py','ply_data.py')]
    bindings=[_binding(p) for p in paths]
    output.mkdir(parents=True)
    command=[sys.executable,'-B','-X','utf8','-c',CHILD,str(pipeline),str(local),str(output/'coverage'),args.selection_scope,args.tracking_profile or '',str(sfm_run) if sfm_run else '']
    save(output/'PLAN.json',dict(argv=command,cwd=str(ROOT),implementation=bindings,
        execute=args.execute,threads=2,wall_seconds=300,memory_bytes=2<<30,
        sources=[str(pipeline),str(local)]+([str(sfm_run)] if sfm_run else []),selection_scope=args.selection_scope,
        tracking_profile=args.tracking_profile,source_sfm_override=_binding(sfm_run) if sfm_run else None,
        new_model_loads=0,new_segmentation=0))
    if not args.execute:print('PLANNED');return 0
    entry=_invoke(ROOT,output,'coverage',command,seconds=300,memory=2<<30)
    code=entry_code(entry)
    drift=[r['path'] for r in bindings if _binding(r['path'])!=r]
    from integrations.demo_pipeline.parallel_photo_audit import read_json
    path=output/'coverage/RESULT.json'
    detail=read_json(path) if path.is_file() else None
    okay=code==0 and not drift and detail and detail['source_unchanged'] and detail['status']=='PARTIAL_CONDITIONAL_EXPANSION'
    result=dict(status='PARTIAL_CONDITIONAL_EXPANSION' if okay else 'NO_ACCEPTED_DELIVERY',
        selection_scope=args.selection_scope,tracking_profile=args.tracking_profile,source_sfm_override=str(sfm_run) if sfm_run else None,
        process_code=code,reclaim_confirmed=entry.get('reclaim_confirmed'),implementation_drift=drift,
        geometry_accepted=False,blender={'status':'NOT_RUN'})
    if args.blender and okay:
        from integrations.blender_export import export_blend
        result['blender']=export_blend(output/'coverage/SCENES.json',output/'blender')
    save(output/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False))
    return 0 if okay and (not args.blender or result['blender']['status']=='SUCCESS') else (code or 2)


if __name__=='__main__':raise SystemExit(main())
