"""Plan or execute bounded local foreground reconstruction; never fill holes."""
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

CHILD="""import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from integrations.demo_pipeline.partial_reconstruction import run
r=run(sys.argv[1],sys.argv[2])
print(r['status'])
raise SystemExit(0 if r['status']=='PARTIAL_CONDITIONAL_MODEL' else 2)
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profile', choices=('local-three-view-k0-v1','local-three-view-k0-whole-video-v1','local-three-view-k0-whole-video-dense-v1'), default='local-three-view-k0-v1')
    parser.add_argument('--execute',action='store_true',help='without this switch only freeze a plan')
    parser.add_argument('--blender',action='store_true',help='export supported local pieces with independent readback')
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.partial_reconstruction import prepare,save
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    output=args.output.resolve()
    plan,_=prepare(args.sfm_run, **({'profile_id':args.profile} if args.profile != 'local-three-view-k0-v1' else {}))
    source=Path(plan['image_root'])
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    code_paths=[Path(__file__),*[ROOT/'integrations/demo_pipeline'/n for n in
        ('partial_reconstruction.py','partial_point_gate.py','partial_surface.py',
         'registration_probe.py','parallel_photo_audit.py','pipeline.py','supervision.py')],
        ROOT/'src/kozlov_lab/reconstruction/export.py',ROOT/'integrations/viewer/ply_data.py',
        ROOT/'integrations/viewer/export.py']
    bindings=[_binding(p) for p in code_paths]
    output.mkdir(parents=True,exist_ok=False)
    plan_path=output/'PLAN.json';save(plan_path,plan)
    command=[sys.executable,'-B','-X','utf8','-c',CHILD,str(plan_path),str(output/'local')]
    save(output/'ENTRY_PLAN.json',dict(argv=command,cwd=str(ROOT),implementation=bindings,
        plan_binding=_binding(plan_path),wall_seconds=480,memory_bytes=2<<30,threads=2,
        max_mapping_calls=sum(g['scheduled'] for g in plan['graph']['groups']),
        source_images_reprocessed=False,geometry_accepted=False,execute=args.execute))
    if not args.execute:
        print(json.dumps(dict(status='PLANNED',groups=plan['graph']['groups']),ensure_ascii=False));return 0
    entry=_invoke(ROOT,output,'local_reconstruction',command,seconds=480,memory=2<<30)
    process_code=entry_code(entry)
    drift=[]
    for ref in plan['source_bindings']+bindings:
        try:
            if _binding(ref['path'])!=ref:drift.append(ref['path'])
        except OSError:drift.append(ref['path'])
    detail=None
    path=output/'local/RESULT.json'
    if path.is_file():
        from integrations.demo_pipeline.parallel_photo_audit import read_json
        detail=read_json(path)
    deliverable=bool(process_code==0 and not drift and detail and detail['source_unchanged']
        and detail['status']=='PARTIAL_CONDITIONAL_MODEL' and detail['scenes'])
    result=dict(status='PARTIAL_CONDITIONAL_MODEL' if deliverable else 'NO_ACCEPTED_DELIVERY',
        process_code=process_code,reclaim_confirmed=entry.get('reclaim_confirmed'),source_drift=drift,
        report=str(output/'local/index.html') if detail else None,geometry_accepted=False,
        scale_status='unknown',blender={'status':'NOT_RUN'})
    if args.blender and deliverable:
        from integrations.blender_export import export_blend
        result['blender']=export_blend(output/'local/SCENES.json',output/'blender')
    save(output/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if deliverable and (not args.blender or result['blender']['status']=='SUCCESS') else (process_code or 2)


if __name__=='__main__':raise SystemExit(main())
