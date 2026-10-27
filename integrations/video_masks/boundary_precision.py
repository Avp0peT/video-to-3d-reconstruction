"""Bound mask-boundary uncertainty, without segmentation or binary-mask repair."""
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw


@dataclass(frozen=True)
class BoundaryPolicy:
    margin_pixels: float = 3.0
    min_source_iou: float = .8
    max_frames: int = 4096
    max_pixels: int = 16_777_216

    def __post_init__(self):
        for key in ('margin_pixels','min_source_iou'):
            x=getattr(self,key)
            if type(x) not in (int,float) or not math.isfinite(x):raise ValueError('finite policy required')
        if self.margin_pixels<0 or not 0<=self.min_source_iou<=1:raise ValueError('invalid policy range')
        for key,maximum in (('max_frames',4096),('max_pixels',16_777_216)):
            x=getattr(self,key)
            if type(x) is not int or not 1<=x<=maximum:raise ValueError('bounded integer policy required')


def _mask(value,shape=None):
    value=np.asarray(value)
    if value.dtype!=np.uint8 or value.ndim!=2 or not value.size or (shape is not None and value.shape!=shape) or not np.isin(value,[0,255]).all():
        raise ValueError('same-grid binary uint8 mask required')
    return value


def derive_trimap(union,base,supports,base_root,policy=BoundaryPolicy(),cv=None):
    """Select one alternative per root, then distinguish agreement from truth.

    Definite background is conservative with respect to the unchanged union:
    distance from any foreground includes all foreground admitted by that union.
    """
    if not isinstance(policy,BoundaryPolicy):raise ValueError('BoundaryPolicy required')
    union=_mask(union);base=_mask(base,union.shape)
    if union.size>policy.max_pixels:raise ValueError('pixel budget exceeded')
    if type(base_root) is not int or base_root<0:raise ValueError('integer base root required')
    if np.any((base==255)&(union==0)):raise ValueError('union excludes base foreground')
    grouped={};rejected=[]
    candidates=[dict(mask=base,root_seed_decode_index=base_root,source='base'),*supports]
    for item in candidates:
        root=item['root_seed_decode_index']
        if type(root) is not int or root<0:raise ValueError('integer source root required')
        mask=_mask(item['mask'],union.shape)
        if np.any((mask==255)&(union==0)):raise ValueError('source foreground outside supplied union')
        denominator=int(((mask!=0)|(base!=0)).sum())
        iou=float(((mask!=0)&(base!=0)).sum()/denominator) if denominator else 1.0
        digest=hashlib.sha256(mask.tobytes()).hexdigest()
        row=dict(root_seed_decode_index=root,iou_with_base=iou,mask_pixels_sha256=digest,source=item.get('source'))
        if iou<policy.min_source_iou:rejected.append(dict(row,reason='BELOW_SOURCE_IOU'));continue
        grouped.setdefault(root,[]).append((iou,digest,mask,row))
    selected=[max(grouped[root],key=lambda x:(x[0],x[1])) for root in sorted(grouped)]
    if not selected:raise ValueError('no admitted source root')
    if cv is None:
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv([])
    if cv is None:raise RuntimeError('existing CPU OpenCV unavailable')
    fg=union==255
    to_bg=cv.distanceTransform(np.pad(fg.astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
    # Padding with zero makes the image exterior unknown, rather than deep background.
    to_any_fg=cv.distanceTransform(np.pad((~fg).astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
    all_fg=np.logical_and.reduce([x[2]==255 for x in selected])
    all_bg=np.logical_and.reduce([x[2]==0 for x in selected])
    known_fg=fg&(to_bg>policy.margin_pixels)&all_fg
    known_bg=(~fg)&all_bg&(to_any_fg>policy.margin_pixels)
    trimap=np.full(union.shape,128,np.uint8);trimap[known_bg]=0;trimap[known_fg]=255
    validity=np.uint8(trimap!=128)*255
    return dict(trimap=trimap,validity=validity,
        stats=dict(pixels=union.size,union_foreground_pixels=int(fg.sum()),known_foreground_pixels=int(known_fg.sum()),
                   known_background_pixels=int(known_bg.sum()),unknown_pixels=int((trimap==128).sum()),
                   unknown_foreground_pixels=int(((trimap==128)&fg).sum()),
                   disagreement_pixels=int((~all_fg&~all_bg).sum()),admitted_roots=len(selected),
                   source_status='SINGLE_SOURCE_NOT_TRUTH' if len(selected)==1 else 'MULTI_ROOT_AGREEMENT_NOT_TRUTH'),
        selected_sources=[x[3] for x in selected],rejected_sources=rejected,
        same_root_alternatives_not_votes=True)


def _read(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def _save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def _binding(path,expected=None):
    path=Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size>128<<20:raise ValueError('bounded regular input required')
    blob=path.read_bytes();record=dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
    if expected is not None and record['sha256']!=expected:raise ValueError('input SHA256 mismatch: '+str(path))
    return record


def prepare(source,output,indices=None,policy=BoundaryPolicy()):
    """Freeze selected existing frame references before CPU processing."""
    source=Path(source).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('fresh disjoint output required')
    refs={}
    def bind(path,expected=None):
        r=_binding(path,expected);refs[r['path']]=r;return Path(r['path'])
    docs={name:_read(bind(source/name)) for name in ('RESULT.json','ASSEMBLY.json','FUSION.json')}
    if docs['RESULT.json'].get('mode')!='union' or docs['RESULT.json'].get('status')!='FULL_TIMELINE_FUSED_EXPORTED':
        raise ValueError('completed literal-union source required')
    assembly={r['decode_index']:r for r in docs['ASSEMBLY.json']['rows']}
    fusion={r['decode_index']:r for r in docs['FUSION.json']['rows']}
    if len(assembly)!=len(docs['ASSEMBLY.json']['rows']) or len(fusion)!=len(docs['FUSION.json']['rows']) or set(assembly)!=set(fusion):
        raise ValueError('duplicate or mismatched source frame domains')
    selected=sorted(assembly) if indices is None else sorted(indices)
    if (not selected or len(selected)>policy.max_frames or any(type(i) is not int or i<0 for i in selected)
            or len(set(selected))!=len(selected) or not set(selected)<=set(assembly)):
        raise ValueError('unique bounded existing frame indices required')
    frames=[]
    for index in selected:
        a,f=assembly[index],fusion[index]
        if f['mode']!='union':raise ValueError('mixed fusion mode refused')
        base=bind(f['base_mask_path'],f['base_mask_sha256'])
        candidate=bind(a['mask_path'],a['mask_sha256'])
        supports=[]
        for x in f['support_evidence']:
            path=bind(x['mask_path'],x['mask_sha256'])
            supports.append(dict(path=str(path),sha256=x['mask_sha256'],root_seed_decode_index=x['root_seed_decode_index']))
        image=bind(source/'frames'/f'{index:06d}.jpg')
        frames.append(dict(decode_index=index,base=str(base),union=str(candidate),image=str(image),
                           base_root=a['selected_root_seed_decode_index'],supports=supports))
    code_root=Path(__file__).resolve().parents[2]
    for path in (Path(__file__),code_root/'scripts/review_mask_precision.py',code_root/'integrations/demo_pipeline/temporal_review.py'):bind(path)
    out.mkdir(parents=True)
    plan=dict(schema='mask-boundary-review-plan-v1',policy=asdict(policy),source_run=str(source),frames=frames,
              source_bindings=list(refs.values()),published_for_reconstruction=False,
              label_domain='TRIMAP_0_BACKGROUND_128_UNKNOWN_255_FOREGROUND_DIAGNOSTIC_ONLY',
              maximum_cpu_executions=1,new_model_executions=0,new_gpu_executions=0,installations=0)
    _save(out/'PLAN.json',plan);return plan


def execute(output):
    out=Path(output).resolve(strict=True)
    if (out/'START.json').exists():raise ValueError('review already attempted')
    plan=_read(out/'PLAN.json');refs=list(plan['source_bindings'])+[_binding(out/'PLAN.json')]
    def unchanged():
        if any(_binding(r['path'])!=r for r in refs):raise ValueError('bound input drift')
    unchanged();_save(out/'START.json',dict(cpu_execution_count=1,new_model_executions=0,new_gpu_executions=0,installations=0))
    started=time.monotonic();rows=[];policy=BoundaryPolicy(**plan['policy'])
    from integrations.demo_pipeline.temporal_review import load_cv
    cv=load_cv(refs)
    if cv is None:raise RuntimeError('bound existing CPU OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    for name in ('union_upper_bound','trimaps','validity','overlays'):(out/name).mkdir()
    def mask(path):
        with Image.open(path) as image:
            if image.mode!='L':raise ValueError('L mask required')
            return _mask(np.asarray(image).copy())
    try:
        for frame in plan['frames']:
            if time.monotonic()-started>240:raise TimeoutError('CPU review240s budget exceeded')
            union,base=mask(frame['union']),mask(frame['base'])
            supports=[dict(mask=mask(x['path']),root_seed_decode_index=x['root_seed_decode_index'],source=x['path']) for x in frame['supports']]
            result=derive_trimap(union,base,supports,frame['base_root'],policy,cv)
            with Image.open(frame['image']) as im:
                image=im.convert('RGB')
            if image.size!=(union.shape[1],union.shape[0]):raise ValueError('source RGB/mask grid mismatch')
            name=f'{frame["decode_index"]:06d}.png'
            shutil.copyfile(frame['union'],out/'union_upper_bound'/name)
            Image.fromarray(result['trimap']).save(out/'trimaps'/name)
            Image.fromarray(result['validity']).save(out/'validity'/name)
            for folder,expected in [('trimaps',result['trimap']),('validity',result['validity'])]:
                if not np.array_equal(np.asarray(Image.open(out/folder/name)),expected):raise ValueError('PNG readback mismatch')
            if (out/'union_upper_bound'/name).read_bytes()!=Path(frame['union']).read_bytes():raise ValueError('union copy drift')
            pixels=np.asarray(image).copy();unknown=result['trimap']==128;fg=result['trimap']==255
            pixels[unknown]=(.45*pixels[unknown]+.55*np.array([255,190,20])).astype(np.uint8)
            pixels[fg]=(.75*pixels[fg]+.25*np.array([30,210,255])).astype(np.uint8)
            overlay=Image.new('RGB',(image.width,image.height+48),(20,26,35));overlay.paste(Image.fromarray(pixels),(0,48))
            ImageDraw.Draw(overlay).text((8,8),'CYAN: source-agreed core (not truth)\nAMBER: UNKNOWN / unchanged background',fill='white')
            overlay.save(out/'overlays'/name)
            rows.append(dict(decode_index=frame['decode_index'],stats=result['stats'],selected_sources=result['selected_sources'],
                             rejected_sources=result['rejected_sources'],union_diff_pixels=0,overlay='overlays/'+name))
        unchanged()
        result=dict(schema='mask-boundary-review-v1',status='BOUNDARY_UNCERTAINTY_REVIEWED_NOT_SEGMENTATION_ACCEPTED',
                    frames=rows,policy=plan['policy'],source_bindings=refs,source_unchanged=True,published_for_reconstruction=False,
                    binary_training_export=False,mesh_arguments_published=False,new_model_executions=0,new_gpu_executions=0,
                    cpu_execution_count=1,installations=0,elapsed_seconds=time.monotonic()-started,
                    summary=dict(frames=len(rows),unknown_pixels=sum(r['stats']['unknown_pixels'] for r in rows),
                                 unknown_foreground_pixels=sum(r['stats']['unknown_foreground_pixels'] for r in rows),
                                 union_diff_pixels=0),
                    limitations=['Source agreement and fixed core are not manually verified object truth.',
                                 'No RGB refinement, SAM/GrabCut execution or binary-mask change.',
                                 'UNKNOWN must not be interpreted as background or sent to old binary mask consumers.',
                                 'Union upper bound retains existing false positives; supplied validity is diagnostic only.'])
        _save(out/'RESULT.json',result)
        cards=''.join('<h2>Frame '+str(r['decode_index'])+'</h2><img style="max-width:576px" src="'+r['overlay']+'">' for r in rows)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Boundary uncertainty / not accepted segmentation</h1><p>Unchanged union, source-agreed core, UNKNOWN boundary/disagreement. Not published for reconstruction.</p><a href="RESULT.json">Evidence</a>'+cards,encoding='utf8')
        return result
    except Exception as exc:
        _save(out/'RESULT.json',dict(schema='mask-boundary-review-v1',status='FAILED',error=type(exc).__name__+': '+str(exc),
                                    published_for_reconstruction=False,cpu_execution_count=1,new_model_executions=0,new_gpu_executions=0))
        raise
