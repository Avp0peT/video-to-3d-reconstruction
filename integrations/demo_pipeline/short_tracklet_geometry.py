"""Saved-pose geometry admission for preserved short tracks; never fit cameras."""
from collections import Counter
from copy import deepcopy
import numpy as np
from .original_anchor_review import _camera, _numeric

RIGHTS=dict(geometry_accepted=False,new_model_points=0,new_cameras=0,native_injected=False)


def assess_tracklets(ledger, frames, masks, camera_result, *, budget=lambda:None):
    """Use only observations actually in this prefix, in one saved model.

    No historical observations are silently appended. Missing intermediate
    poses are not approximated by an endpoint pose. At most 32 eligible groups
    are evaluated in frozen track/model order under the caller's shared clock.
    """
    if len(frames)!=len(masks):raise ValueError('FRAME_MASK_COUNT')
    if not ledger['records']:
        return dict(status='SHORT_TRACKLET_GEOMETRY_LEDGER_COMPLETE',records=[],counts={},
            numeric_groups=0,max_numeric_groups=32,pose_interpolation=False,history_augmented=False,**RIGHTS)
    bydecode={};models={}
    for f,m in zip(frames,masks):
        if type(f.get('decode_index')) is not int or f['decode_index'] in bydecode:
            raise ValueError('UNIQUE_DECODE_IDENTITY_REQUIRED')
        bydecode[f['decode_index']]=(f,m)
    for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        key=model['model_key']
        if key in models:raise ValueError('DUPLICATE_MODEL')
        cams={}
        for table in ('cameras','unsupported_cameras'):
            for index,c in model.get(table,{}).items():
                if isinstance(index,str) and index.isdigit():index=int(index)
                if type(index) is not int or index in cams or c.get('input_index')!=index:
                    raise ValueError('CAMERA_INPUT_IDENTITY_MISMATCH')
                cams[index]=c
        models[key]=cams
    out=[];seen=set();evaluated=0
    for r in ledger['records']:
        budget()
        tid=r['tracklet_id']
        if tid in seen:raise ValueError('DUPLICATE_TRACKLET')
        seen.add(tid)
        record=dict(tracklet_id=tid,status='EXCLUDED_SHORT_PREFIX',model_assessments=[],**RIGHTS)
        out.append(record)
        if r['status']!='CANDIDATE_2D_TRACKLET':continue
        obs=r['observations']
        if len(obs)<3 or len({o['decode_index'] for o in obs})!=len(obs):
            raise ValueError('CANDIDATE_PREFIX_SCOPE_INVALID')
        for key,cams in sorted(models.items()):
            present=[o for o in obs if o['decode_index'] in bydecode and bydecode[o['decode_index']][0]['input_index'] in cams]
            if not present:continue
            g=dict(model_key=key,status='BLOCKED_MISSING_POSES',camera_views=len(present),usable_views=0,
                missing_camera_decode_indices=[],unknown_mask_decode_indices=[],observations=[],
                admitted_observations=[],fit=None,evaluation=None,observability=None,
                leave_one_view_out=[],radial_domains=[],strict_radial_domain_check=[],raw_predictions=[],
                branch_scope='same saved model; existing central radial branch diagnostic only',**RIGHTS)
            record['model_assessments'].append(g);image_ids=set()
            for o in obs:
                budget();idx=o['decode_index'];pair=bydecode.get(idx)
                f,m=pair if pair is not None else (None,None)
                c=cams.get(f['input_index']) if f is not None else None
                row=dict(decode_index=idx,raw_xy=o['raw_xy'],input_index=f['input_index'] if f else None,
                    camera_present=c is not None,mask_status='UNKNOWN_NOT_SEGMENTED',reason=None)
                g['observations'].append(row)
                if c is None:g['missing_camera_decode_indices'].append(idx)
                mask_ok=False
                if f is not None:
                    row['mask_status']=f.get('original_mask_status','UNKNOWN')
                    if isinstance(m,np.ndarray) and m.dtype==np.uint8 and m.shape==(f['height'],f['width']) and f.get('original_mask_status')=='OK':
                        if not np.isin(m,(0,255)).all():raise ValueError('NONBINARY_MASK')
                        x,y=np.floor(o['raw_xy']).astype(int)
                        mask_ok=bool(0<=x<f['width'] and 0<=y<f['height'] and m[y,x]==255)
                if not mask_ok:g['unknown_mask_decode_indices'].append(idx)
                if c is None:row['reason']='NO_SAVED_CAMERA';continue
                if (o['width'],o['height'])!=(f['width'],f['height']):raise ValueError('TRACKLET_FRAME_DIMENSION_MISMATCH')
                if c.get('camera_model')!='SIMPLE_RADIAL':row['reason']='UNSUPPORTED_CAMERA_MODEL';continue
                _camera(c,f,f['input_index'])
                if c['image_id'] in image_ids:raise ValueError('DUPLICATE_CAMERA_OBSERVATION')
                image_ids.add(c['image_id'])
                if not mask_ok:row['reason']='NO_CONFIRMED_FOREGROUND';continue
                g['admitted_observations'].append(dict(input_index=f['input_index'],image_id=c['image_id'],
                    raw_pixels=deepcopy(o['raw_xy']),params=c['params'],R=c['R'],t=c['t'],
                    width=f['width'],height=f['height']))
            g['usable_views']=len(g['admitted_observations'])
            if g['camera_views']<2:continue
            if g['usable_views']<2:g['status']='BLOCKED_FOREGROUND_OR_CAMERA_MODEL';continue
            if evaluated>=32:g['status']='NOT_EVALUATED_GROUP_BUDGET';continue
            evaluated+=1;g['status']='EVALUATING';g['loo_status']='NOT_EVALUATED_TWO_VIEWS'
            _numeric(g,budget)
            g['evidence_level']='MULTIVIEW_CONDITIONAL_DIAGNOSTIC' if g['usable_views']>=3 else 'TWO_VIEW_PROVISIONAL_DIAGNOSTIC'
            g['diagnostic_scope']='saved estimated cameras, shared selected observations; no independent identity or geometry acceptance'
        statuses=[g['status'] for g in record['model_assessments']]
        record['status']=('NO_SAVED_MODEL_FOR_PREFIX' if not statuses else
            'CONDITIONAL_GEOMETRY_EVALUATED' if any(g.get('fit') is not None for g in record['model_assessments']) else
            'BLOCKED_MISSING_POSES' if all(s=='BLOCKED_MISSING_POSES' for s in statuses) else
            'GEOMETRY_NOT_EVALUATED')
    return dict(status='SHORT_TRACKLET_GEOMETRY_LEDGER_COMPLETE',records=out,
        counts=dict(Counter(r['status'] for r in out)),numeric_groups=evaluated,max_numeric_groups=32,
        pose_interpolation=False,history_augmented=False,**RIGHTS)
