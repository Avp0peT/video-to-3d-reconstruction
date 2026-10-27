"""Bounded direct-versus-intermediate LK evidence, without geometry acceptance."""
from copy import deepcopy
from collections import Counter
import numpy as np
from .seed_pair_validation import _need, _xy, _output, _safe

PROFILE = dict(profile_id='intermediate-tracking-v1', max_gap=8, max_points=64,
               win_size=21, max_level=3, max_iterations=30, epsilon=.01,
               max_lk_error=25., max_fb_px=1., max_cycle_px=1.)
RIGHTS = dict(geometry_accepted=False, identity_verified=False,
              intermediate_masks_verified=False, native_injected=False,
              original_observations_modified=False)


def compare_edge(edge, decoded, cv, *, budget=lambda: None, progress=lambda e: None):
    """Compare identical source seeds. Raw pixel centers equal LK coordinates + .5.

    Input validation raises ValueError. Cancellation/budget and native CV errors
    propagate; partial_result on the exception and progress preserve prior evidence.
    Unknown intermediate masks never establish foreground membership.
    """
    path=edge['decode_path']
    _need(isinstance(path,list) and 2<=len(path)<=9 and all(type(i) is int and i>=0 for i in path),'INVALID_DECODE_PATH')
    direction=1 if path[-1]>path[0] else -1
    _need(all(b-a==direction for a,b in zip(path,path[1:])),'NONCONSECUTIVE_DECODE_PATH')
    _need(all(type(edge.get(k)) is int for k in ('source_decode_index','target_decode_index')) and path[0]==edge['source_decode_index'] and path[-1]==edge['target_decode_index'],'DECODE_ENDPOINT_MISMATCH')
    _need(isinstance(edge.get('model_key'),str) and bool(edge['model_key']),'INVALID_MODEL_KEY')
    _need(all(type(edge.get(k)) is int and edge[k]>=0 for k in ('source_input_index','target_input_index')) and edge['source_input_index']!=edge['target_input_index'],'INVALID_INPUT_INDEX')
    points=edge['points']
    _need(isinstance(points,list) and 1<=len(points)<=64,'POINT_BOUND')
    ids=set();shape=None
    for i in path:
        f=decoded[i];w=f['width'];h=f['height']
        _need(type(w) is int and type(h) is int and w>0 and h>0,'INVALID_DIMENSIONS')
        gray=f['gray'];mask=f['mask']
        _need(isinstance(gray,np.ndarray) and gray.dtype==np.uint8 and gray.shape==(h,w),'INVALID_GRAY')
        _need(shape is None or gray.shape==shape,'FRAME_SHAPE_MISMATCH');shape=gray.shape
        _need(isinstance(f.get('path'),str) and bool(f['path']),'MISSING_FRAME_PATH')
        if i in (path[0],path[-1]):
            _need(f.get('original_mask_status')=='OK' and isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(h,w),'ORIGINAL_MASK_REQUIRED')
            _need(np.isin(mask,(0,255)).all(),'NONBINARY_ORIGINAL_MASK')
        else:
            _need(mask is None and f.get('original_mask_status')=='UNKNOWN_NOT_SEGMENTED','INTERMEDIATE_MASK_MUST_BE_UNKNOWN')
    for point in points:
        gid=point.get('group_id');_need(isinstance(gid,str) and gid and gid not in ids,'INVALID_OR_DUPLICATE_GROUP_ID');ids.add(gid)
        _xy(point['source_raw_xy'])
    result=dict(status='RUNNING',profile=deepcopy(PROFILE),model_key=edge['model_key'],
                source_input_index=edge['source_input_index'],target_input_index=edge['target_input_index'],
                decode_path=list(path),rows=[],counts={},flow_calls=0,**RIGHTS)
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,30,.01))

    def inside(raw,index):
        p=_xy(raw)-.5;f=decoded[index]
        _need(0<=p[0]<=f['width']-1 and 0<=p[1]<=f['height']-1,'POINT_OUTSIDE_IMAGE')
        if f['mask'] is not None:
            x,y=np.floor(raw).astype(int)
            _need(0<=x<f['width'] and 0<=y<f['height'] and f['mask'][y,x]==255,'POINT_OUTSIDE_ORIGINAL_MASK')

    def flow(a,b,raw,record):
        inside(raw,a)
        budget();result['flow_calls']+=1
        xy,status,error=cv.calcOpticalFlowPyrLK(decoded[a]['gray'].copy(),decoded[b]['gray'].copy(),
            (_xy(raw)-.5).astype(np.float32).reshape(1,1,2).copy(),None,**lk)
        record.update(from_decode_index=a,to_decode_index=b,input_raw_xy=list(raw),
                      xy=_safe(xy),status=_safe(status),error=_safe(error),
                      destination_mask_status=decoded[b]['original_mask_status'])
        output(xy,(1,1,2),'LK_XY');output(status,(1,1),'LK_STATUS',True);output(error,(1,1),'LK_ERROR')
        measured=(xy[0,0].astype(float)+.5).tolist();record['observed_raw_xy']=measured
        numerical_need(float(error[0,0])<=25,'LK_ERROR_THRESHOLD');inside(measured,b)
        return measured

    def arm(route,source,out):
        current=list(source)
        try:
            inside(current,route[0]);out['observed_raw_coordinates'].append(dict(decode_index=route[0],raw_xy=list(current)))
            for a,b in zip(route,route[1:]):
                step=dict(forward={},backward={});out['steps'].append(step)
                # Only local numerical rejection is caught; budgets/native calls remain outside catch.
                measured=flow(a,b,current,step['forward'])
                out['observed_raw_coordinates'].append(dict(decode_index=b,raw_xy=list(measured)))
                returned=flow(b,a,measured,step['backward'])
                step['fb_error_px']=float(np.linalg.norm(_xy(returned)-_xy(current)))
                numerical_need(step['fb_error_px']<=1,'PER_LEG_FB_THRESHOLD');current=measured
            cycle=out['fullcycle'];reverse=list(reversed(route))
            for a,b in zip(reverse,reverse[1:]):
                record={};cycle['steps'].append(record);current=flow(a,b,current,record)
            cycle['error_px']=float(np.linalg.norm(_xy(current)-_xy(source)))
            numerical_need(cycle['error_px']<=1,'WHOLE_CYCLE_THRESHOLD')
            cycle['status']='PASSED';out['status']='PASSED'
        except _NumericalRejection as exc:
            out['reasons'].append(str(exc))

    # Convert only explicit validation failures inside numerical helpers. Never
    # catch ValueError from a budget hook or calcOpticalFlowPyrLK as rejection.
    original_inside=inside
    def inside(raw,index):
        try:original_inside(raw,index)
        except ValueError as exc:raise _NumericalRejection(str(exc)) from exc
    original_output=_output
    def output(value,shape,label,status=False):
        try:return original_output(value,shape,label,status)
        except ValueError as exc:raise _NumericalRejection(str(exc)) from exc
    # Flow checks use local wrappers; thresholds also use the numerical wrapper.
    def numerical_need(value,reason):
        if not value:raise _NumericalRejection(reason)
    # Numerical wrappers leave preflight validation and external exceptions strict.


    try:
        for point in points:
            row=dict(group_id=point['group_id'],source_raw_xy=deepcopy(point['source_raw_xy']),**RIGHTS)
            result['rows'].append(row)
            for name,route in [('direct',[path[0],path[-1]]),('dense',path)]:
                out=dict(status='REJECTED',reasons=[],steps=[],observed_raw_coordinates=[],fullcycle=dict(status='NOT_COMPLETED',steps=[],error_px=None))
                row[name]=out;arm(route,row['source_raw_xy'],out)
            d=row['direct']['status']=='PASSED';n=row['dense']['status']=='PASSED'
            row['outcome']='BOTH_PASS' if d and n else 'RECOVERED' if n else 'LOST' if d else 'BOTH_FAIL'
            progress(deepcopy(result))
    except Exception as exc:
        result['status']='INTERRUPTED';result['counts']=dict(Counter(r.get('outcome','INCOMPLETE') for r in result['rows']))
        try:exc.partial_result=deepcopy(result)
        except (AttributeError,TypeError):pass
        try:progress(deepcopy(result))
        except Exception:pass
        raise
    result['status']='COMPARED';result['counts']={k:sum(r['outcome']==k for r in result['rows']) for k in ('RECOVERED','LOST','BOTH_PASS','BOTH_FAIL')}
    return result


class _NumericalRejection(Exception):
    pass
