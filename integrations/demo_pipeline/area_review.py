"""Evidence-bound whole-area image alignment, separate from point tracking."""
import json
from pathlib import Path
import time
from .temporal_review import fingerprint
from .area_tracking import track_areas
from .area_tracking_report import render_area_tracking


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def review_area_sequence(plan, decoded, output, refs, cv, *, point_baseline=(), budget=lambda: None):
    budget(); output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    start=time.monotonic(); result=dict(status='RUNNING', regions=[], geometry_accepted=False, new_model_points=0)
    def verify():
        budget()
        for row in refs:
            if fingerprint(row['path']) != row:
                raise ValueError('whole-area source drift: '+row['path'])
    try:
        verify()
        # Selection geometry is fixed before this estimator. No point motion enters it.
        write(output/'PLAN.json', plan)
        result=track_areas(plan, decoded, cv, budget=budget)
        result['measurement_wall_seconds']=time.monotonic()-start
        result['selection_basis']='same frozen root ROIs as point baseline; estimator consumes image pixels, not measured point motion'
        result['point_baseline']=[{k:r[k] for k in ('region_id','center_baseline','expanded_grid','counts')} for r in point_baseline]
        verify()
        artifacts=render_area_tracking(output, result, decoded, budget=budget)
        result['wall_seconds']=time.monotonic()-start
        result['source_unchanged']=True
        write(output/'RESULT.json',result)
        verify()
        artifacts += [output/'PLAN.json',output/'RESULT.json']
        return dict(status=result['status'], counts=result['counts'], regions=len(result['regions']),
            native_calls=result['native_calls'], wall_seconds=result['wall_seconds'],
            geometry_accepted=False,new_model_points=0,
            report='intermediate_frames/population/regions/whole_area/index.html',
            source_bindings=[fingerprint(p) for p in artifacts])
    except Exception as exc:
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',
            reason=f'{type(exc).__name__}: {exc}',source_unchanged=None,wall_seconds=time.monotonic()-start)
        if getattr(exc,'partial_result',None) is not None:
            result['interrupted_measurement']=exc.partial_result
        try:write(output/'FAILURE.json',result)
        except Exception as delivery:
            if hasattr(exc,'add_note'):exc.add_note('Whole-area failure evidence: '+str(delivery))
        raise
