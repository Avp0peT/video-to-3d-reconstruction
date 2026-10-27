"""Object-neutral, bounded reverse tracking and evidence-selected bridge frames."""
from pathlib import Path
import argparse
import html
from integrations.demo_pipeline.pipeline import ROOT, read, save, sha, utc
from . import region_improvement as reverse
from . import region_bridge_experiment as bridge
from .region_bridge_inputs import choose_intermediates as select_intermediates
from .object_region_policy import POLICY
from .region_experiment import drift, resolve_source_geometry


def _code_bindings():
    paths=set(reverse.CODE)
    paths.update('integrations/video_masks/'+p for p in bridge.CODE)
    paths.update(('integrations/video_masks/region_refinement.py','scripts/run_photo_demo.py'))
    return {str(ROOT/p):sha(ROOT/p) for p in sorted(paths)}


def _save_state(out, state):
    save(out/'REFINEMENT.json',state)
    latest=state.get('latest_comparison')
    link=('<p><a href="'+html.escape(latest,quote=True)+'">阶段对比与三维查看</a></p>') if latest else ''
    body=html.escape(str(state.get('error') or state.get('stages',{})))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>通用区域增强</title>'
        '<h1>物体区域自动增强</h1><p>'+html.escape(state['status'])+'</p>'+link+'<pre>'+body+'</pre>'
        '<p>反向跟踪与补帧完成不代表表面、尺寸或几何质量验收。无自动补洞或分量拼接。</p>'
        '<a href="REFINEMENT.json">完整执行记录</a>',encoding='utf-8')


def prepare(source_run, output, source_geometry=None):
    """Freeze a complete regional source; no inference or native mapping."""
    source=Path(source_run).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or source.is_relative_to(out):
        raise FileExistsError('fresh disjoint refinement output required')
    original=read(source/'PLAN.json')
    if original.get('schema')!='glove-region-plan-v1':
        raise ValueError('completed region input required; raw photos or sampled masks need preparation first')
    timeline=original['timeline']
    _,preflight=select_intermediates(timeline,[],POLICY)
    if not 2<=preflight['endpoint_count']<=64:
        raise ValueError('initial region endpoint budget is 2..64; no silent subsampling')
    geometry=resolve_source_geometry(source,source_geometry)
    code=_code_bindings()
    out.mkdir(parents=True)
    state=dict(schema='object-region-refinement-v1',source_run=str(source),source_geometry=geometry,
               created_utc=utc(),status='PREPARING',stages={},policy=dict(POLICY),
               object_category_required=False,endpoint_count=preflight['endpoint_count'],
               geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown',
               automatic_retry=False,maximum_bridge_passes=1,code_bindings=code)
    _save_state(out,state)
    try:
        frozen=reverse.prepare(source,out/'reverse',geometry)
        state['source_bindings']=frozen['source_bindings']
        state['prepared_stage_bindings']={str(out/'reverse/PLAN.json'):sha(out/'reverse/PLAN.json')}
        state['status']='PREPARED'
    except Exception as exc:
        state.update(status='PRECONDITION_FAILED',error=f'{type(exc).__name__}: {exc}')
        _save_state(out,state)
        raise
    _save_state(out,state)
    return state


def execute(output):
    out=Path(output).resolve(strict=True);state=read(out/'REFINEMENT.json')
    if state.get('schema')!='object-region-refinement-v1' or state['status']!='PREPARED':
        raise ValueError('prepared unattempted object refinement required')
    if state['policy']!=POLICY:
        raise ValueError('refinement policy differs from the frozen version')
    if drift(state['source_bindings']) or drift(state['code_bindings']) or drift(state['prepared_stage_bindings']):
        raise ValueError('prelaunch input/code drift')
    source=Path(state['source_run'])
    state.update(status='RUNNING',started_utc=utc());_save_state(out,state)
    try:
        first=reverse.execute(out/'reverse')
        state['stages']['reverse']=first['status'];_save_state(out,state)
        if first['status']!='EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED':
            raise RuntimeError('reverse stage failed; no continuation or retry')
        original=read(source/'PLAN.json')
        result=read(out/'reverse/assessment/ASSESSMENT.json')
        added,selection=select_intermediates(original['timeline'],result['models'],state['policy'])
        state['bridge_selection']=selection
        if not selection.get('resource_limit_satisfied',True):
            raise RuntimeError('bridge resource policy rejected the existing input')
        if added:
            bridge.prepare(out/'reverse',out/'bridge',state['policy'])
            second=bridge.execute(out/'bridge')
            state['stages']['bridge']=second['status']
            if second['status']!='BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED':
                raise RuntimeError('bridge stage failed; no continuation or retry')
            actual=read(out/'bridge/input/PLAN.json')['added_decode_indices']
            if actual!=added:
                raise RuntimeError('bridge prepared selection differs from frozen controller decision')
            state['latest_comparison']='bridge/assessment/index.html'
        else:
            state['stages']['bridge']='NOT_RUN_NO_ELIGIBLE_FRAMES'
            state['latest_comparison']='reverse/assessment/index.html'
        state['status']='REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'
    except Exception as exc:
        state.update(status='PARTIAL_DIAGNOSTIC',error=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(finished_utc=utc(),input_drift=drift(state['source_bindings']),
                     code_drift=drift(state['code_bindings']),child_plan_drift=drift(state['prepared_stage_bindings']))
        if state['input_drift'] or state['code_drift'] or state['child_plan_drift']:state['status']='DRIFT_REFUSED'
        _save_state(out,state)
    return state


def run(source_run,output,source_geometry=None):
    prepare(source_run,output,source_geometry)
    return execute(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--source-geometry');a=p.parse_args(argv)
    result=run(a.source_run,a.run,a.source_geometry)
    return 0 if result['status']=='REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED' else 1
