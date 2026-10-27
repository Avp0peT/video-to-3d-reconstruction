"""Constrained CPU GrabCut experiment; candidates never replace source masks."""
from dataclasses import asdict,dataclass
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .boundary_precision import _binding,_read,_save
from .narrow_gap import detect_narrow_gaps,evaluate_gap_preservation


@dataclass(frozen=True)
class RefinePolicy:
    iterations: int = 2
    threads: int = 1
    rng_seed: int = 0
    max_changed_foreground_fraction: float = .01
    max_pixels: int = 16_777_216
    maximum_seconds: int = 240

    def __post_init__(self):
        if type(self.iterations) is not int or self.iterations!=2 or type(self.threads) is not int or self.threads!=1 or type(self.rng_seed) is not int or self.rng_seed!=0:
            raise ValueError('frozen two-iteration one-thread seed-zero experiment required')
        if type(self.max_changed_foreground_fraction) not in (int,float) or not np.isfinite(self.max_changed_foreground_fraction) or not 0<=self.max_changed_foreground_fraction<=.01:
            raise ValueError('foreground change cap may only tighten one percent')
        for name,maximum in [('max_pixels',16_777_216),('maximum_seconds',240)]:
            x=getattr(self,name)
            if type(x) is not int or not 1<=x<=maximum:raise ValueError('bounded integer policy required')


def _validate(union,trimap):
    union,trimap=np.asarray(union),np.asarray(trimap)
    if (union.dtype!=np.uint8 or union.ndim!=2 or not union.size or not np.isin(union,[0,255]).all()
            or trimap.dtype!=np.uint8 or trimap.shape!=union.shape or not np.isin(trimap,[0,128,255]).all()):
        raise ValueError('same-grid uint8 binary union and three-label trimap required')
    if not union.any() or np.any((trimap==255)&(union==0)) or np.any((trimap==0)&(union!=0)):
        raise ValueError('nonempty union consistent with definite trimap labels required')
    return union,trimap


def _gap_guard(original,candidate,cv):
    """Fail closed against the supplied mask grid, not segmentation ground truth.

    Trimap 128 is label uncertainty; it is not an invalid source-image pixel.
    This entry point has full-grid binary masks and no external validity map.
    """
    scope='FULL_SUPPLIED_MASK_GRID'
    try:
        validity=np.ones(original.shape,bool)
        diagnostics=detect_narrow_gaps(original,validity,cv=cv)
        row=evaluate_gap_preservation(original,candidate,validity,diagnostics,cv=cv)
        if not isinstance(row,dict):raise ValueError('gap guard did not return a record')
        violations=row.get('violations')
        if not isinstance(violations,list) or any(not isinstance(x,str) for x in violations):
            raise ValueError('gap guard violations are not explicit strings')
        maps={}
        for name in ('protected_filled','path_disconnected'):
            value=np.asarray(row.get(name))
            if value.dtype!=np.bool_ or value.shape!=original.shape:
                raise ValueError('gap guard map has invalid domain: '+name)
            maps[name+'_pixels']=int(value.sum())
        if not isinstance(row.get('stats'),dict) or not isinstance(diagnostics.get('stats'),dict):
            raise ValueError('gap guard statistics are missing')
        # Freeze only finite JSON evidence; diagnostic image arrays stay in the kernel.
        stats=json.loads(json.dumps(row['stats'],allow_nan=False))
        source_stats=json.loads(json.dumps(diagnostics['stats'],allow_nan=False))
        verified=(row.get('status')=='GAP_PRESERVATION_PASSED_NOT_GEOMETRY_ACCEPTED'
                  and row.get('ok') is True and row.get('preserved') is True
                  and not violations and not any(maps.values()))
        rejected=(row.get('status')=='GAP_PRESERVATION_REJECTED'
                  and row.get('ok') is False and row.get('preserved') is False)
        if not verified and not rejected:raise ValueError('gap guard result is unknown or inconsistent')
        return dict(status=row['status'],ok=verified,preserved=verified,
                    violations=list(violations),stats=stats,source_diagnostic_stats=source_stats,
                    validity_scope=scope,trimap_unknown_is_invalid=False,
                    geometry_accepted=False,segmentation_accepted=False,**maps)
    except Exception as exc:
        return dict(status='GAP_PRESERVATION_UNVERIFIED',ok=False,preserved=False,
                    violations=['GAP_GUARD_UNVERIFIABLE'],validity_scope=scope,
                    trimap_unknown_is_invalid=False,geometry_accepted=False,
                    segmentation_accepted=False,error=type(exc).__name__+': '+str(exc))


def evaluate_proposal(union,trimap,proposal,policy=RefinePolicy(),cv=None):
    union,trimap=_validate(union,trimap);proposal=np.asarray(proposal)
    if proposal.dtype!=np.uint8 or proposal.shape!=union.shape or not np.isin(proposal,[0,255]).all():
        raise ValueError('finite same-grid binary candidate required')
    if cv is None:
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv([])
    if cv is None:raise RuntimeError('existing CPU OpenCV unavailable')
    original=union!=0;candidate=proposal!=0;changed=original!=candidate;reasons=[]
    if np.any(candidate&~original):reasons.append('BACKGROUND_OR_HOLE_ADDITION')
    if np.any(changed&(trimap!=128)):reasons.append('KNOWN_TRIMAP_LABEL_CHANGED')
    foreground=int(original.sum());count=int(changed.sum())
    if count>foreground*policy.max_changed_foreground_fraction:reasons.append('FOREGROUND_CHANGE_CAP_EXCEEDED')
    components,labels,stats,_=cv.connectedComponentsWithStats(original.astype(np.uint8),connectivity=8)
    retained=np.bincount(labels[candidate],minlength=components)
    lost=[i for i in range(1,components) if retained[i]==0]
    if lost:reasons.append('ORIGINAL_COMPONENT_ELIMINATED')
    gap_guard=_gap_guard(original,candidate,cv)
    if not gap_guard['ok']:reasons.append('GAP_PRESERVATION_NOT_VERIFIED')
    return dict(status='REFUSED' if reasons else 'BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED',
                reasons=reasons,changed_pixels=count,removed_pixels=int((original&~candidate).sum()),
                added_pixels=int((candidate&~original).sum()),change_fraction_of_original_foreground=count/foreground,
                original_foreground_components=components-1,eliminated_component_ids=lost,
                core_unchanged=not bool(np.any(changed&(trimap==255))),
                known_background_unchanged=not bool(np.any(changed&(trimap==0))),
                unknown_pixels=int((trimap==128).sum()),unknown_is_background=False,
                gap_guard=gap_guard)


def refine_candidate(rgb,union,trimap,policy=RefinePolicy(),cv=None):
    if not isinstance(policy,RefinePolicy):raise ValueError('RefinePolicy required')
    union,trimap=_validate(union,trimap);rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.shape!=(*union.shape,3) or union.size>policy.max_pixels:
        raise ValueError('bounded exact-grid RGB uint8 required')
    if cv is None:
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv([])
    if cv is None:raise RuntimeError('existing CPU OpenCV unavailable')
    cv.setNumThreads(1);cv.ocl.setUseOpenCL(False);cv.setRNGSeed(0)
    initial=np.where(union!=0,cv.GC_PR_FGD,cv.GC_PR_BGD).astype(np.uint8)
    initial[trimap==255]=cv.GC_FGD;initial[trimap==0]=cv.GC_BGD
    if not np.any(trimap==255) or not np.any(trimap==0):
        return dict(proposal=union.copy(),status='REFUSED',reasons=['INSUFFICIENT_DEFINITE_FOREGROUND_OR_BACKGROUND'],grabcut_calls=0,
                    gap_guard=dict(status='NOT_EVALUATED',ok=False,preserved=False,violations=['CANDIDATE_NOT_EVALUATED']))
    bg_model=np.zeros((1,65),np.float64);fg_model=np.zeros((1,65),np.float64)
    try:
        cv.grabCut(rgb[:,:,::-1].copy(),initial,None,bg_model,fg_model,2,cv.GC_INIT_WITH_MASK)
    except cv.error as exc:
        return dict(proposal=union.copy(),status='REFUSED',reasons=['GRABCUT_ERROR: '+str(exc)],grabcut_calls=1,
                    gap_guard=dict(status='NOT_EVALUATED',ok=False,preserved=False,violations=['CANDIDATE_NOT_EVALUATED']))
    # This projection enforces the no-addition contract; UNKNOWN remains a prior,
    # not definite background. Definite-label mutations are still independently checked.
    proposed=np.uint8(((initial==cv.GC_FGD)|(initial==cv.GC_PR_FGD))&(union!=0))*255
    return dict(evaluate_proposal(union,trimap,proposed,policy,cv),proposal=proposed,grabcut_calls=1)


def prepare(review,output,indices=None,policy=RefinePolicy()):
    review=Path(review).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out.is_relative_to(review) or review.is_relative_to(out):raise ValueError('fresh disjoint refinement output required')
    refs={}
    def add(path,expected=None):
        r=_binding(path,expected)
        if r['path'] in refs and refs[r['path']]!=r:raise ValueError('input drift while preparing')
        refs[r['path']]=r;return Path(r['path'])
    plan=_read(add(review/'PLAN.json'));result=_read(add(review/'RESULT.json'))
    if result.get('schema')!='mask-boundary-review-v1' or result.get('source_unchanged') is not True or result.get('published_for_reconstruction') is not False:
        raise ValueError('completed diagnostic trimap source required')
    declared={r['path']:r for r in plan['source_bindings']}
    for record in plan['source_bindings']:add(record['path'],record['sha256'])
    by_id={f['decode_index']:f for f in plan['frames']}
    chosen=sorted(by_id) if indices is None else sorted(indices)
    if not chosen or len(set(chosen))!=len(chosen) or any(type(i) is not int or i<0 for i in chosen) or not set(chosen)<=set(by_id):
        raise ValueError('unique declared source indices required')
    frames=[]
    for i in chosen:
        row=by_id[i];image=Path(row['image']).resolve(strict=True)
        if str(image) not in declared:raise ValueError('RGB has no explicit input manifest binding')
        add(image,declared[str(image)]['sha256']);union=add(review/'union_upper_bound'/f'{i:06d}.png')
        original=add(row['union'])
        if union.read_bytes()!=original.read_bytes():raise ValueError('review union no longer exact source mask')
        trimap=add(review/'trimaps'/f'{i:06d}.png')
        frames.append(dict(decode_index=i,image=str(image),union=str(union),trimap=str(trimap)))
    code_root=Path(__file__).resolve().parents[2]
    for path in (Path(__file__),Path(__file__).with_name('boundary_precision.py'),Path(__file__).with_name('narrow_gap.py'),code_root/'scripts/refine_mask_boundary.py',code_root/'integrations/demo_pipeline/temporal_review.py'):add(path)
    out.mkdir(parents=True)
    frozen=dict(schema='boundary-refinement-plan-v1',review_run=str(review),frames=frames,policy=asdict(policy),
                source_bindings=list(refs.values()),maximum_cpu_executions=1,published_for_reconstruction=False,
                batch_refusal_action='ALL_DEFAULT_MASKS_REMAIN_ORIGINAL_UNION',
                default_mask_policy='ALWAYS_UNCHANGED_ORIGINAL_UNION',proposal_promotion=False,
                gap_guard_validity='FULL_SUPPLIED_MASK_GRID')
    _save(out/'PLAN.json',frozen);return frozen


def execute(output):
    out=Path(output).resolve(strict=True)
    if (out/'START.json').exists():raise ValueError('CPU experiment already attempted')
    plan=_read(out/'PLAN.json');refs=list(plan['source_bindings'])+[_binding(out/'PLAN.json')]
    def unchanged():
        if any(_binding(r['path'])!=r for r in refs):raise ValueError('bound source drift')
    unchanged();start=time.monotonic()
    _save(out/'START.json',dict(cpu_execution_count=1,new_sam_executions=0,new_gpu_executions=0,installations=0,
                              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    policy=RefinePolicy(**plan['policy']);rows=[];all_pass=True;calls=0
    from integrations.demo_pipeline.temporal_review import load_cv
    cv=load_cv(refs)
    if cv is None:raise RuntimeError('existing bound CPU OpenCV unavailable')
    for directory in ('proposals','default_masks','overlays'):(out/directory).mkdir()
    # Freeze safe source bytes before backend work. A late input drift must never
    # cause defaults to be restored from the now-untrusted current source path.
    declared={r['path']:r for r in refs};safe_defaults={}
    for frame in plan['frames']:
        path=Path(frame['union']);blob=path.read_bytes();expected=declared[str(path)]
        if len(blob)!=expected['bytes'] or hashlib.sha256(blob).hexdigest()!=expected['sha256']:
            raise ValueError('source union drift before safe copy')
        safe_defaults[out/'default_masks'/f'{frame["decode_index"]:06d}.png']=blob
    def restore_safe_defaults():
        for path,blob in safe_defaults.items():
            if not path.exists() or path.read_bytes()!=blob:path.write_bytes(blob)
    restore_safe_defaults()
    try:
        for frame in plan['frames']:
            if time.monotonic()-start>policy.maximum_seconds:raise TimeoutError('CPU experiment240s limit')
            with Image.open(frame['image']) as im:
                if im.mode!='RGB':raise ValueError('bound image must be RGB')
                image=im.copy();rgb=np.asarray(image).copy()
            with Image.open(frame['union']) as im:union=np.asarray(im).copy()
            with Image.open(frame['trimap']) as im:trimap=np.asarray(im).copy()
            candidate=refine_candidate(rgb,union,trimap,policy,cv);proposal=candidate.pop('proposal')
            calls+=candidate['grabcut_calls'];all_pass &= candidate['status']=='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED'
            name=f'{frame["decode_index"]:06d}.png';Image.fromarray(proposal).save(out/'proposals'/name)
            if not np.array_equal(np.asarray(Image.open(out/'proposals'/name)),proposal):raise ValueError('proposal PNG readback mismatch')
            canvas=Image.new('RGB',(image.width*4,image.height+65),(20,26,35));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
            removed=(union!=0)&(proposal==0)
            for column,kind in enumerate(('SOURCE RGB','UNCHANGED UNION','PROPOSED / '+candidate['status'],'REMOVED PIXELS (MAGENTA)')):
                pixels=rgb.copy()
                if column in (1,2):
                    hit=(union if column==1 else proposal)!=0
                    pixels[hit]=(.65*pixels[hit]+.35*np.array([0,210,255])).astype(np.uint8)
                if column==3:pixels[removed]=[255,0,180]
                canvas.paste(Image.fromarray(pixels),(column*image.width,65));ImageDraw.Draw(canvas).text((column*image.width+5,8),kind,font=font,fill='white')
            canvas.save(out/'overlays'/name)
            rows.append(dict(decode_index=frame['decode_index'],**candidate,proposal='proposals/'+name,
                             overlay='overlays/'+name,default_mask='default_masks/'+name))
        for row in rows:
            row['default_kind']='UNCHANGED_ORIGINAL_UNION'
        unchanged()
        restore_safe_defaults()
        result=dict(schema='boundary-refinement-v1',status='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED' if all_pass else 'REFUSED',
                    frames=rows,policy=plan['policy'],source_bindings=refs,source_unchanged=True,published_for_reconstruction=False,
                    geometry_accepted=False,segmentation_accepted=False,cpu_execution_count=1,grabcut_calls=calls,
                    new_sam_executions=0,new_gpu_executions=0,installations=0,elapsed_seconds=time.monotonic()-start,
                    batch_refusal_action=plan['batch_refusal_action'],
                    default_mask_policy='ALWAYS_UNCHANGED_ORIGINAL_UNION',proposal_promotion=False,
                    limitations=['Automatic trimap experiment with fixed GrabCut, not independent segmentation truth.',
                                 'Colour agreement does not establish glove or object identity; UNKNOWN was not definite background.',
                                 'No original mask, holes or reconstruction arguments are changed; candidates cannot replace originals.'])
        _save(out/'RESULT.json',result)
        cards=''.join('<h2>Frame '+str(r['decode_index'])+' / '+r['status']+'</h2><img style="max-width:100%" src="'+r['overlay']+'">' for r in rows)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Boundary refinement experiment / '+result['status']+'</h1><p>Default masks always remain the original union. Unaccepted proposals are separate and never promoted. No reconstruction publication.</p><a href="RESULT.json">Evidence</a>'+cards,encoding='utf8')
        return result
    except Exception as exc:
        restore_safe_defaults()
        _save(out/'RESULT.json',dict(schema='boundary-refinement-v1',status='REFUSED',error=type(exc).__name__+': '+str(exc),
                    source_unchanged=False,published_for_reconstruction=False,cpu_execution_count=1,grabcut_calls=calls,
                    new_sam_executions=0,new_gpu_executions=0,installations=0,
                    default_mask_policy='ALWAYS_UNCHANGED_ORIGINAL_UNION',proposal_promotion=False))
        raise
