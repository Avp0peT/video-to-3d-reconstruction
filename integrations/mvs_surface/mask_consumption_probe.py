"""Independent dense-only mask assay, deliberately outside production admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from .runner import (ROOT, binding, plan_files, save_json, stage_inputs, commands,
                     make_plan, _effective_dense_config)
from .supervision import run_supervised
from .scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap
from integrations.video_masks.face_mask_adapter import verify_depth

ARMS = ('DISABLED', 'RECT', 'BLACK')
TARGET = 'image_00000080.png'


def need(ok, message):
    if not ok: raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resize_nearest(mask, shape):
    """OpenCV INTER_NEAREST centre-index convention (floor, not PIL nearest)."""
    mask=np.asarray(mask);h,w=shape
    need(mask.ndim==2 and mask.dtype==np.bool_ and h>0 and w>0,'bool mask and positive shape required')
    y=np.minimum(np.arange(h)*mask.shape[0]//h,mask.shape[0]-1)
    x=np.minimum(np.arange(w)*mask.shape[1]//w,mask.shape[1]-1)
    return mask[y[:,None],x[None,:]]


def erode_two(mask):
    a=np.pad(np.asarray(mask,bool),2,constant_values=False)
    h,w=mask.shape
    return np.logical_and.reduce([a[y:y+h,x:x+w] for y in range(5) for x in range(5)])


def authenticate_depth(d, camera):
    """Authenticate embedded identity/KRC, including legitimate resized K."""
    w,h=d.depth_size;ow,oh=camera['width'],camera['height']
    need(0<w<=ow and 0<h<=oh and d.image_size==d.depth_size,'unsupported depth dimensions')
    scaled=deepcopy(camera);scaled.update(width=w,height=h)
    k=np.asarray(camera['K'],float).copy();sx,sy=w/ow,h/oh
    k[0,:]*=sx;k[1,:]*=sy
    k[0,2]+=(sx-1)*.5;k[1,2]+=(sy-1)*.5
    scaled['K']=k
    verify_depth(d,scaled)


def find_depth(work, camera, check=lambda:None):
    found=[];records=[]
    paths=sorted(Path(work).glob('depth*.dmap'));need(len(paths)<=256,'excess depth files')
    for path in paths:
        check();d=read_dmap(path);records.append(binding(path,512<<20))
        if Path(d.image_name_utf8.replace('\\','/')).name==Path(camera['name']).name:
            authenticate_depth(d,camera);found.append(d)
    need(len(found)<=1,'duplicate embedded target depth identity')
    return (found[0] if found else None),records


def depth_counts(depth, forbidden):
    if depth is None: return dict(available=False,positive_total=None,positive_inside=None,positive_outside=None)
    d=np.asarray(depth);need(d.ndim==2 and d.dtype.kind in 'fiu','numeric depth raster required')
    inside=erode_two(resize_nearest(forbidden,d.shape))
    outside=erode_two(~resize_nearest(forbidden,d.shape))
    positive=np.isfinite(d)&(d>0)
    return dict(available=True,positive_total=int(positive.sum()),positive_inside=int((positive&inside).sum()),
                positive_outside=int((positive&outside).sum()),inside_pixels=int(inside.sum()),outside_pixels=int(outside.sum()))


def verdict(rows):
    """Missing dmaps/reclamation are UNKNOWN; measured directional violations FAIL."""
    if any(a not in rows or not rows[a].get('reclaimed') or not rows[a].get('counts',{}).get('available')
           or not rows[a].get('execution_ok') for a in ARMS):
        return 'UNKNOWN'
    d,r,b=[rows[a]['counts'] for a in ARMS]
    if d['positive_inside']<50 or r['positive_outside']<50: return 'UNKNOWN'
    if r['positive_inside']!=0 or b['positive_total']!=0: return 'FAIL'
    return 'REFERENCE_MASK_CONSUMPTION_VERIFIED_NOT_GEOMETRY_ACCEPTANCE'


def unchanged(records):
    for rec in records: need(binding(rec['path'],512<<20)==rec,'frozen file drift: '+rec['path'])


def code_records():
    return [binding(ROOT/p) for p in ('integrations/mvs_surface/mask_consumption_probe.py',
        'scripts/probe_mvs_mask_consumption.py','integrations/openmvs_depth/dmap.py',
        'integrations/video_masks/face_mask_adapter.py','integrations/cad_delivery/supervise.py')]


def prepare(source_plan, output, baseline_depth_work=None):
    start=time.monotonic();path=Path(source_plan).resolve(strict=True);out=Path(output).resolve()
    need(not out.exists() and not out.is_relative_to(path.parent) and not path.parent.is_relative_to(out),'fresh disjoint probe output required')
    plan=load(path)
    need(plan==make_plan(**plan['arguments']),'source production plan changed')
    need(len(plan['frames'])==12,'frozen 12-view assay required')
    frame=next((f for f in plan['frames'] if Path(f['image']['name']).name==TARGET),None)
    need(frame is not None,'target absent')
    cam=frame['image'];refs=[binding(path),*plan_files(plan),*code_records(),binding(Path(__import__('sys').executable))]
    depth_work=Path(baseline_depth_work).resolve(strict=True) if baseline_depth_work else path.parent/'work'
    need(not out.is_relative_to(depth_work) and not depth_work.is_relative_to(out),'probe output overlaps baseline depth')
    d,ds=find_depth(depth_work,cam);refs+=ds
    need(d is not None,'source target dmap required to freeze observable rectangle')
    h,w=cam['height'],cam['width'];rect=np.zeros((h,w),bool)
    rect[h//4:3*h//4,w//4:3*w//4]=True
    observed=depth_counts(d.depth,rect)
    need(observed['positive_inside']>=50,'central fixed rectangle has insufficient old observations; no tuning')
    out.mkdir(parents=True);Image.fromarray(rect.astype(np.uint8)*255).save(out/'forbidden.png')
    arms=[]
    for name in ARMS:
        work=out/name/'work';work.mkdir(parents=True)
        stage_inputs(plan,work,time.monotonic()+120)
        for f in plan['frames']:
            mask=np.full((f['image']['height'],f['image']['width']),255,np.uint8)
            if Path(f['image']['name']).name==TARGET and name!='DISABLED':
                if name=='BLACK': mask[:]=0
                else: mask[rect]=0
            # DISABLED carries the identical RECT input: only ignore label differs.
            if Path(f['image']['name']).name==TARGET and name=='DISABLED': mask[rect]=0
            Image.fromarray(mask).save(work/'__mvs_masks'/f['mask_name'])
        assay=deepcopy(plan);assay['arguments']['ignore_label']=-1 if name=='DISABLED' else 0
        argv=commands(assay,work)[0][1]
        staged=[binding(p,512<<20) for p in sorted(work.rglob('*')) if p.is_file()]
        arms.append(dict(name=name,work=str(work),argv=argv,staged=staged,ignore_label=assay['arguments']['ignore_label']))
    refs.append(binding(out/'forbidden.png'));unchanged(refs)
    doc=dict(schema='native-reference-mask-assay-v1',status='PREPARED_NOT_EXECUTED',
        scope='Independent diagnostic. Mutated all-white/black masks are NOT admitted production plans.',
        source_plan=str(path),baseline_depth_work=str(depth_work),camera=cam,rectangle=[w//4,h//4,3*w//4,3*h//4],source_observations=observed,
        refs=refs,arms=arms,budget=dict(total_seconds=600,arm_seconds=180,threads=2,memory_bytes=8<<30,output_bytes=1<<30),
        elapsed_seconds=time.monotonic()-start)
    save_json(out/'PREPARED.json',doc,exclusive=True)
    return binding(out/'PREPARED.json')


def execute(prepared, prepared_sha256):
    path=Path(prepared).resolve(strict=True);out=path.parent
    need(binding(path)['sha256']==prepared_sha256,'prepared record digest mismatch')
    doc=load(path);need(doc['schema']=='native-reference-mask-assay-v1','wrong probe schema')
    # Exclusive marker seals the only attempt even if validation or execution fails.
    save_json(out/'EXECUTION_ATTEMPT.json',dict(prepared=binding(path)),exclusive=True)
    start=time.monotonic();deadline=start+600;rows={};result=dict(status='UNKNOWN',arms=rows,geometry_accepted=False,new_native_launch_attempts=0)
    def check():
        if time.monotonic()>deadline: raise TimeoutError('600s assay deadline')
    try:
        unchanged(doc['refs']);need(all(r in doc['refs'] for r in code_records()),'probe code changed')
        source=load(doc['source_plan']);cam=doc['camera']
        need([a['name'] for a in doc['arms']]==list(ARMS),'fixed assay arm order required')
        with Image.open(out/'forbidden.png') as im:rect=np.array(im)!=0
        save_json(out/'BINDINGS_PRE.json',doc['refs'])
        for arm in doc['arms']:
            check();work=Path(arm['work']);unchanged(arm['staged'])
            need(work==out/arm['name']/'work','arm work directory differs')
            need({str(p.resolve()) for p in work.rglob('*') if p.is_file()}=={r['path'] for r in arm['staged']},'unexpected files/cache in prepared work')
            expected=deepcopy(source);expected['arguments']['ignore_label']=-1 if arm['name']=='DISABLED' else 0
            need(arm['argv']==commands(expected,work)[0][1],'dense-only arm argv differs')
            need(not list(work.glob('depth*.dmap')) and not (work/'__mvs_dense.mvs').exists(),'old dense cache forbidden')
            evidence=work/'__probe_runtime';evidence.mkdir()
            row=dict(reclaimed=False,execution_ok=False,counts=depth_counts(None,rect));rows[arm['name']]=row
            result['new_native_launch_attempts']+=1;save_json(out/'RESULT.json',result)
            native=run_supervised(arm['argv'],cwd=work,output=evidence,wall_seconds=min(180,deadline-time.monotonic()),
                memory_bytes=8<<30,output_bytes=1<<30,label=arm['name'])
            save_json(evidence/'supervisor_result.json',native,exclusive=True)
            row.update(supervisor=native,reclaimed=native.get('reclaim_confirmed') is True)
            row['execution_ok']=(native.get('status')=='completed' and native.get('root_exit_code')==0
                and row['reclaimed'] and type(native.get('pid')) is int and native['pid']>0
                and native.get('argv')==arm['argv'] and native.get('cwd')==str(work)
                and not native.get('errors') and not native.get('log_errors'))
            need(row['reclaimed'],'native reclamation unconfirmed; subsequent arms forbidden')
            d,records=find_depth(work,cam,check);row['depth_bindings']=records
            row['counts']=depth_counts(None if d is None else d.depth,rect)
            if row['execution_ok']:
                row['dense_config_effective']=_effective_dense_config(source,work,evidence)
            # Source + effective masks must remain fixed; native cfg rewrites are expected.
            unchanged([r for r in arm['staged'] if Path(r['path']).name not in ('__mvs_dense.cfg',)])
            if d is not None:
                inside=erode_two(resize_nearest(rect,d.depth.shape));valid=np.isfinite(d.depth)&(d.depth>0)
                image=np.zeros((*valid.shape,3),np.uint8);image[valid]=[50,180,230];image[inside]=[90,40,40];image[inside&valid]=[255,40,40]
                Image.fromarray(image).save(out/(arm['name']+'_depth.png'))
            save_json(out/'RESULT.json',result)
            if not row['execution_ok']: break  # Failed execution is not retried; assay is incomplete.
        result['status']=verdict(rows)
        unchanged(doc['refs']);save_json(out/'BINDINGS_POST.json',doc['refs']);result['source_unchanged']=True
    except Exception as exc:
        result.update(status='UNKNOWN',error=type(exc).__name__+': '+str(exc))
    after=[]
    for ref in doc['refs']:
        try:
            actual=binding(ref['path'],512<<20);after.append(dict(expected=ref,actual=actual,unchanged=actual==ref))
        except Exception as exc:
            after.append(dict(expected=ref,unchanged=False,error=type(exc).__name__+': '+str(exc)))
    result['source_unchanged']=all(r['unchanged'] for r in after)
    if not result['source_unchanged']:result['status']='UNKNOWN'
    save_json(out/'BINDINGS_POST.json',after)
    result.update(elapsed_seconds=time.monotonic()-start,limitations=[
        'Reference depth-estimation mask behaviour only; target NCC patch masking is not established.',
        'Missing dmaps, insufficient disabled observations or failed supervision are UNKNOWN.',
        'No mesh, geometry adoption or prior state replacement.'])
    save_json(out/'RESULT.json',result)
    cards=''.join('<h2>'+a+'</h2><img style="max-width:100%" src="'+a+'_depth.png">' for a in ARMS if (out/(a+'_depth.png')).exists())
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Reference-mask assay: '+result['status']+'</h1><a href="RESULT.json">Counts, logs and bindings</a><p>Blue: positive depth. Dark red: forbidden interior. Bright red: positive depth inside forbidden interior.</p>'+cards,encoding='utf-8')
    return result
