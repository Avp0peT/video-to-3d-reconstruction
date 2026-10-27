"""One bounded density-policy reconstruction on the exact existing frame set."""
from __future__ import annotations
from copy import deepcopy
import argparse
import html
from pathlib import Path

from integrations.demo_pipeline.pipeline import ROOT,LIGHT_REL,_invoke,entry_code,read,save,sha,utc
from integrations.demo_pipeline.balanced_point_run import binding,check_bindings
from .density_sampling import default_policy
from .track_identity import default_policy as default_identity_policy


def frame_cap(plan):
    from .region_geometry import _expanded_policy
    count=sum(r.get('input_index') is not None for r in plan['timeline'])
    return 96 if count>64 or _expanded_policy(plan.get('endpoint_expansion_policy',{})) else 64


def derived_plan(original):
    if original.get('schema')!='glove-region-plan-v1':
        raise ValueError('regional timeline plan required')
    timeline=original['timeline']
    indices=[r['decode_index'] for r in timeline]
    if (not indices or any(type(i) is not int for i in indices)
            or indices!=list(range(indices[0],indices[-1]+1))):
        raise ValueError('complete consecutive existing timeline required')
    endpoints=[r['input_index'] for r in timeline if r.get('input_index') is not None]
    if (not 2<=len(endpoints)<=96 or any(type(i) is not int or i<0 for i in endpoints)
            or any(a>=b for a,b in zip(endpoints,endpoints[1:]))):
        raise ValueError('unique ordered 2..96 existing endpoints required')
    from .region_geometry import _validate_frame_limit
    _validate_frame_limit(len(endpoints),frame_cap(original),original)
    result=deepcopy(original)
    result['density_policy']=default_policy()
    result['identity_policy']=default_identity_policy()
    return result


def render(out,state):
    items=[]
    for stage in state['stages']:
        items.append('<li>'+html.escape(stage['name']+' : '+stage['status'])+'</li>')
    links=[]
    for label,path in [('新模型与逐步结果','surface/index.html'),('新网格','surface/mesh_viewer.html'),
                       ('新稠密点云','surface/dense_viewer.html'),('Blender','surface/blender/model.blend'),
                       ('局部密度复核','surface/point_density/index.html'),('新旧对照','comparison/index.html')]:
        if (out/path).is_file():links.append('<li><a href="'+path+'">'+label+'</a></li>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>密度优先重建</title>'
        '<style>body{font:18px system-ui;max-width:1050px;margin:40px auto;background:#f4f6f8}li{margin:12px}pre{white-space:pre-wrap}</style>'
        '<h1>同一视频、同一掩膜、同一端点：密度优先建模</h1><p>'+html.escape(state['status'])+'</p>'
        '<p>新候选重新匹配、三角化后才进入模型。各分量独立，孔洞保留。完整性、物理尺度及精度未验收。</p>'
        '<ul>'+''.join(items)+'</ul><ul>'+''.join(links)+'</ul><pre>'+html.escape(state.get('error',''))+'</pre>'
        '<p><a href="PLAN.json">冻结计划</a> · <a href="RESULT.json">执行记录</a></p>',encoding='utf-8')


def prepare(source_run,output):
    source=Path(source_run).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('fresh disjoint output under repository outputs required')
    original=read(source/'input/PLAN.json');child=derived_plan(original)
    declared=original['source_bindings']
    refs=[]
    for path,expected in declared.items():
        record=binding(Path(path))
        if record['sha256']!=expected:raise ValueError('input drift: '+path)
        refs.append(record)
    refs += [binding(source/'input/PLAN.json'),binding(source/'geometry/RESULT.json')]
    sfm=Path(original['baseline_sfm']).resolve(strict=True)
    refs += [binding(p) for p in sorted(sfm.rglob('*')) if p.is_file()]
    from .region_surface import code_bindings
    code=code_bindings()
    for rel in ('integrations/video_masks/density_reconstruction.py','scripts/reconstruct_density.py',
                'integrations/video_masks/region_geometry.py','integrations/video_masks/region_tracks.py',
                'integrations/video_masks/object_region_policy.py','integrations/video_masks/track_identity.py'):
        code.append(binding(ROOT/rel))
    if check_bindings(refs+code):raise ValueError('preparation drift')
    (out/'reconstruction/input').mkdir(parents=True)
    save(out/'reconstruction/input/PLAN.json',child)
    refs.append(binding(out/'reconstruction/input/PLAN.json'))
    plan=dict(schema='density-reconstruction-v1',source_run=str(source),created_utc=utc(),
        timeline_frames=len(child['timeline']),endpoint_count=sum(r.get('input_index') is not None for r in child['timeline']),
        baseline_sfm=str(sfm),source_bindings=refs,code_bindings=code,
        intervention='Add frozen density and target identity policies; preserve exact timeline, masks, native endpoints and SIFT baseline.',
        density_policy=child['density_policy'],identity_policy=child['identity_policy'],new_segmentation=False,automatic_retry=False,
        tracking_seconds=600,geometry_seconds=600,surface_controller_seconds=2400,
        cpu_threads=2,tracking_geometry_commit_bytes=4<<30,surface_outer_commit_bytes=10<<30,
        max_frames=frame_cap(child),geometry_accepted=False)
    save(out/'PLAN.json',plan)
    state=dict(schema='density-reconstruction-result-v1',status='PREPARED',stages=[],
        geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(out/'RESULT.json',state);render(out,state)
    return plan


def execute(output):
    out=Path(output).resolve(strict=True);plan=read(out/'PLAN.json');state=read(out/'RESULT.json')
    if state['status']!='PREPARED' or (out/'logs').exists():raise ValueError('already attempted')
    refs=plan['source_bindings']+plan['code_bindings']+[binding(out/'PLAN.json')]
    if check_bindings(refs):raise ValueError('prelaunch drift')
    python=ROOT/LIGHT_REL;script=ROOT/'scripts/reconstruct_density.py'
    stages=[('tracking',[script,'tracking-worker','--output',out],600,4<<30),
            ('geometry',[script,'geometry-worker','--output',out],600,4<<30),
            ('surface',[ROOT/'scripts/build_region_surface.py','run','--source',out/'reconstruction',
                        '--output',out/'surface'],2400,10<<30)]
    state.update(status='RUNNING',started_utc=utc())
    try:
        for name,args,seconds,memory in stages:
            if check_bindings(refs):raise ValueError('bound source or code changed')
            argv=[str(python),'-B','-X','utf8',*map(str,args)]
            stage=dict(name=name,status='RUNNING',argv=argv,started_utc=utc())
            state['stages'].append(stage);save(out/'RESULT.json',state);render(out,state)
            print('STAGE_BEGIN '+name,flush=True)
            entry=_invoke(ROOT,out,name,argv,seconds=seconds,memory=memory)
            code=entry_code(entry)
            stage.update(exit_code=code,reclaim_confirmed=entry.get('reclaim_confirmed'),finished_utc=utc(),
                         status='COMPLETED' if code==0 else 'FAILED')
            print('STAGE_END '+name+' rc='+str(code),flush=True)
            if code!=0:raise RuntimeError(name+' failed; no automatic retry')
            if name=='tracking':
                track=read(out/'reconstruction/tracking/REGION_TRACKS.json')
                if track.get('density_policy')!=plan['density_policy']:raise ValueError('density path not executed')
                if track.get('identity_policy')!=plan.get('identity_policy'):raise ValueError('identity isolation path not executed')
                state['tracking_metrics']=track['metrics']
            elif name=='geometry':
                geometry=read(out/'reconstruction/geometry/RESULT.json')
                if not geometry.get('models') or geometry.get('source_unchanged') is not True:
                    raise ValueError('new model absent or input drift')
                state['models']=geometry['models']
            else:
                surf=read(out/'surface/RESULT.json')
                if surf.get('status')!='SURFACE_GENERATED_NOT_ACCEPTED':raise ValueError('surface not generated')
                state['surface_status']=surf['status']
            # Subsequent stages cannot silently consume changed stage products.
            folder=out/'reconstruction'/name if name in ('tracking','geometry') else out/'surface'
            refs += [binding(p) for p in sorted(folder.rglob('*')) if p.is_file()]
            save(out/'STAGE_BINDINGS.json',refs);save(out/'RESULT.json',state);render(out,state)
        state['status']='SURFACE_GENERATED_NOT_ACCEPTED'
    except Exception as exc:
        state.update(status='PARTIAL_OR_FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        state['binding_drift']=check_bindings(refs)
        if state['binding_drift']:state['status']='INPUT_DRIFT_REFUSED'
        state['finished_utc']=utc();save(out/'RESULT.json',state);render(out,state)
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','prepare','execute','tracking-worker','geometry-worker'])
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action in ('run','prepare'):
        if args.source is None:parser.error('--source required')
        prepare(args.source,args.output)
        if args.action=='prepare':return 0
    if args.action in ('run','execute'):
        return 0 if execute(args.output)['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
    plan=read(args.output/'PLAN.json')
    if check_bindings(plan['source_bindings']+plan['code_bindings']):raise ValueError('worker prelaunch drift')
    if args.action=='tracking-worker':
        from .region_tracks import run
        run(args.output/'reconstruction/input/PLAN.json',args.output/'reconstruction/tracking')
    else:
        from .region_geometry import run
        run(plan['baseline_sfm'],args.output/'reconstruction/tracking/REGION_TRACKS.json',
            args.output/'reconstruction/geometry',max_frames=plan['max_frames'])
    return 0
