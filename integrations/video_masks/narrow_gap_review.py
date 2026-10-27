"""Bound same-frame fusion replay with explicit narrow-background preservation.

This is a separate candidate; it does not modify a segmentation source, promote
UNKNOWN to background, or publish a reconstruction manifest.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import html
from io import BytesIO
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .boundary_precision import _binding, _read, _save
from .narrow_gap import GapPolicyNew, detect_narrow_gaps, evaluate_gap_preservation

ROOT=Path(__file__).resolve().parents[2]


def _need(value,message):
    if not value:raise ValueError(message)


def _index_rows(rows):
    _need(isinstance(rows,list),'frame array required')
    ids=[r['decode_index'] for r in rows]
    _need(all(type(i) is int and i>=0 for i in ids) and len(set(ids))==len(ids),'unique integer frame IDs required')
    return {r['decode_index']:r for r in rows}


def prepare(source,output,indices=None,policy=GapPolicyNew()):
    source=Path(source).resolve(strict=True);out=Path(output).resolve()
    _need(not out.exists() and out.is_relative_to(ROOT/'outputs') and not out.is_relative_to(source)
        and not source.is_relative_to(out),'fresh disjoint repository output required')
    refs={}
    def add(p,expected=None):
        ref=_binding(p,expected);refs[ref['path']]=ref;return Path(ref['path'])
    result=_read(add(source/'RESULT.json'));assembly=_read(add(source/'ASSEMBLY.json'))
    fusion=_read(add(source/'FUSION.json'));source_plan=_read(add(source/'PLAN.json'))
    separate=(result.get('status')=='FULL_TIMELINE_FUSED_EXPORTED' and result.get('mode')=='union')
    inline=(result.get('status')=='FULL_TIMELINE_EXPORTED' and source_plan.get('mask_fusion_mode')=='union'
        and result.get('fusion',{}).get('mode')=='union')
    _need((separate or inline) and not result.get('input_drift') and not result.get('code_drift'),
        'completed literal-union input required')
    a,f=_index_rows(assembly['rows']),_index_rows(fusion['rows'])
    decode_root=Path(result['source_run']) if separate else source
    decode=_index_rows(_read(add(decode_root/'DECODE.json'))['frames'])
    _need(set(a)==set(f)==set(decode) and len(a)==source_plan['frame_count'],'source frame domains differ')
    chosen=sorted(a) if indices is None else sorted(indices)
    _need(0<len(chosen)<=256 and len(set(chosen))==len(chosen) and
        all(type(i) is int and i in a for i in chosen),'at most256 unique source indices per CPU batch')
    width,height=source_plan['width'],source_plan['height']
    _need(type(width) is int and type(height) is int and 0<width*height<=4_194_304,'bounded image domain')
    frames=[]
    for index in chosen:
        item=f[index];row=a[index]
        _need(item['mode']=='union' and row.get('fusion')==item,'frame fusion ledger mismatch')
        base=add(item['base_mask_path'],item['base_mask_sha256']);union=add(row['mask_path'],row['mask_sha256'])
        original_image=add(decode[index]['path'],decode[index]['sha256'])
        image=add(source/'frames'/f'{index:06d}.jpg',decode[index]['sha256'])
        supports=[]
        for support in item['support_evidence']:
            path=add(support['mask_path'],support['mask_sha256'])
            root=support['root_seed_decode_index'];_need(type(root) is int and root>=0,'integer evidence root')
            supports.append(dict(path=str(path),sha256=support['mask_sha256'],root=root))
        frames.append(dict(decode_index=index,base=str(base),union=str(union),image=str(image),
            original_image=str(original_image),supports=supports,source_status=row['status'],
            original_sam3_status=row['original_status'],base_root=row['selected_root_seed_decode_index']))
    for p in (Path(__file__),Path(__file__).with_name('narrow_gap.py'),Path(__file__).with_name('boundary_precision.py'),
        ROOT/'scripts/protect_narrow_gaps.py',ROOT/'integrations/demo_pipeline/temporal_review.py'):add(p)
    out.mkdir(parents=True)
    plan=dict(schema='narrow-gap-review-plan-v1',source_run=str(source),frames=frames,width=width,height=height,
        policy=asdict(policy),source_bindings=list(refs.values()),maximum_seconds=240,max_frame_count=256,
        maximum_changed_fraction=.01,maximum_cpu_executions=1,new_model_executions=0,
        default_masks='UNCHANGED_LITERAL_UNION',published_for_reconstruction=False,
        scope='Per-frame source-background preservation, not physical or temporal identity proof')
    _save(out/'PLAN.json',plan);return plan


def protect_union(base,literal,valid,diagnostics,cv=None):
    """Block only new foreground; uncertain closure falls back to the whole base."""
    base,literal,valid=[np.asarray(x) for x in (base,literal,valid)]
    _need(base.dtype==literal.dtype==valid.dtype==np.bool_ and base.ndim==2
        and base.shape==literal.shape==valid.shape and not np.any(base&~literal),'bound bool source <= union required')
    # Validate the supplied diagnostics against actual source bytes before use.
    original_guard=evaluate_gap_preservation(base,literal,valid,diagnostics,cv=cv)
    candidate=literal & ~diagnostics['protected_background']
    guard=evaluate_gap_preservation(base,candidate,valid,diagnostics,cv=cv)
    fallback=None
    if not guard['preserved']:
        # A mouth can close outside the narrow-band classification. Do not invent
        # a new path or remove source foreground to force it open.
        candidate=base.copy();fallback='UNION_CONNECTIVITY_CONFLICT_RESTORE_BASE_CANDIDATE'
        guard=evaluate_gap_preservation(base,candidate,valid,diagnostics,cv=cv)
    _need(guard['preserved'] and not np.any(base&~candidate) and not np.any(candidate&~literal),
        'preservation or subset invariant failed')
    return candidate,dict(before=original_guard,after=guard,fallback_reason=fallback)


def _json_guard(guard):
    return {k:v for k,v in guard.items() if not isinstance(v,np.ndarray)}


def execute(output):
    out=Path(output).resolve(strict=True);plan=_read(out/'PLAN.json')
    _need(not (out/'START.json').exists(),'CPU entry already claimed')
    refs=list(plan['source_bindings'])+[_binding(out/'PLAN.json')]
    def unchanged():
        _need(all(_binding(r['path'])==r for r in refs),'bound input/code drift')
    unchanged();_save(out/'START.json',dict(cpu_execution_count=1,new_model_executions=0,new_gpu_executions=0))
    start=time.monotonic();rows=[];policy=GapPolicyNew(**plan['policy'])
    result=dict(schema='narrow-gap-review-v1',status='RUNNING',source_unchanged=False,
        segmentation_accepted=False,geometry_accepted=False,published_for_reconstruction=False,
        cpu_execution_count=1,new_model_executions=0,new_gpu_executions=0,new_mvs_executions=0)
    def check():
        if time.monotonic()-start>plan['maximum_seconds']:raise TimeoutError('240s narrow-gap batch limit')
    from integrations.demo_pipeline.temporal_review import load_cv
    cv=load_cv(refs);_need(cv is not None,'existing CPU OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    for name in ('proposals','default_masks','arrays','overlays','previews'):(out/name).mkdir()
    by_path={r['path']:r for r in refs};safe={}
    def read_mask(path):
        blob=Path(path).read_bytes();ref=by_path[str(path)]
        _need(len(blob)==ref['bytes'] and hashlib.sha256(blob).hexdigest()==ref['sha256'],'mask changed before decode')
        with Image.open(BytesIO(blob)) as image:
            _need(image.mode=='L' and image.size==(plan['width'],plan['height']),'mask L/grid mismatch')
            array=np.asarray(image).copy()
        _need(np.isin(array,[0,255]).all(),'binary labels required')
        return array!=0,blob
    try:
        for frame in plan['frames']:
            _,blob=read_mask(frame['union']);safe[frame['decode_index']]=blob
            (out/'default_masks'/f'{frame["decode_index"]:06d}.png').write_bytes(blob)
        for frame in plan['frames']:
            check();index=frame['decode_index'];base,_=read_mask(frame['base']);literal,_=read_mask(frame['union'])
            recomputed=base.copy()
            for support in frame['supports']:
                sm,_=read_mask(support['path']);recomputed|=sm
            _need(np.array_equal(recomputed,literal),'literal union does not reproduce source evidence')
            valid=np.ones_like(base,bool)
            diag=detect_narrow_gaps(base,valid,policy=policy,cv=cv,check=check)
            candidate,guards=protect_union(base,literal,valid,diag,cv=cv)
            changed=int(np.count_nonzero(candidate!=literal));cap=plan['maximum_changed_fraction']*max(1,int(literal.sum()))
            refused=changed>cap
            name=f'{index:06d}.png';Image.fromarray(candidate.astype(np.uint8)*255).save(out/'proposals'/name)
            with Image.open(out/'proposals'/name) as image:
                _need(image.mode=='L' and np.array_equal(np.asarray(image)!=0,candidate),'proposal readback failed')
            arrays={k:v for k,v in diag.items() if isinstance(v,np.ndarray)}
            arrays.update(before_protected_filled=guards['before']['protected_filled'],
                before_path_disconnected=guards['before']['path_disconnected'])
            np.savez_compressed(out/'arrays'/f'{index:06d}.npz',**arrays)
            with Image.open(frame['image']) as im:
                _need(im.mode=='RGB' and im.size==(plan['width'],plan['height']),'RGB grid mismatch')
                rgb=np.asarray(im).copy()
            h,w=base.shape;canvas=Image.new('RGB',(w*3,h+80),(18,25,35));draw=ImageDraw.Draw(canvas)
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
            for col,label in enumerate(('BASE: NARROW GAPS','LITERAL UNION: FILL CONFLICT','PROTECTED UNION CANDIDATE')):
                pixels=rgb.copy()
                fg=base if col==0 else literal if col==1 else candidate
                pixels[fg]=(.7*pixels[fg]+.3*np.array([30,170,255])).astype(np.uint8)
                if col==0:
                    pixels[diag['narrow_external']]=[30,240,100]
                    pixels[diag['subresolution'] & diag['narrow_external']]=[255,190,20]
                    pixels[diag['enclosed']]=[160,80,255]
                if col==1:pixels[guards['before']['protected_filled'] | guards['before']['path_disconnected']]=[255,20,160]
                if col==2:pixels[literal & ~candidate]=[255,220,20]
                canvas.paste(Image.fromarray(pixels),(col*w,80));draw.text((col*w+8,10),f'FRAME {index} | {label}',font=font,fill='white')
            draw.text((8,40),'Green: open narrow background | Amber: unresolved fine gap | Purple: enclosed | Yellow: blocked addition',font=font,fill='white')
            canvas.save(out/'overlays'/name)
            finite=diag['width_upper_bound'][diag['narrow_external']]
            finite=finite[np.isfinite(finite)]
            row=dict(decode_index=index,status='REFUSED_CHANGE_CAP' if refused else 'GAP_PROTECTED_FUSION_CANDIDATE',
                source_status=frame['source_status'],original_sam3_status=frame['original_sam3_status'],
                source_roots=sorted(set([frame['base_root']]+[r['root'] for r in frame['supports']])),
                roots_are_accuracy_votes=False,stats=diag['stats'],width_bank_center_span_quantiles=
                    np.percentile(finite,[0,25,50,75,100]).tolist() if len(finite) else None,
                blocked_union_additions=changed,base_foreground_lost=int(np.count_nonzero(base & ~candidate)),
                added_foreground_vs_union=int(np.count_nonzero(candidate & ~literal)),
                before_guard=_json_guard(guards['before']),after_guard=_json_guard(guards['after']),fallback_reason=guards['fallback_reason'],
                default_kind='UNCHANGED_LITERAL_UNION',proposal='proposals/'+name,overlay='overlays/'+name,
                interpretation='Source-background preservation only. No independent mask truth or geometry publication.')
            rows.append(row);_save(out/'PROGRESS.json',dict(processed_frames=len(rows),last_index=index,elapsed_seconds=time.monotonic()-start))
            print('NARROW_GAP_FRAME',index,'blocked',changed,'fallback',guards['fallback_reason'],flush=True)
        # Show neighboring frames individually; no unproved cross-frame point or
        # gap identity is inferred from fixed image coordinates.
        groups=[]
        for row in rows:
            if not groups or row['decode_index']!=groups[-1][-1]['decode_index']+1:groups.append([])
            groups[-1].append(row)
        animations=[]
        for group in groups:
            if len(group)<2:continue
            frames=[]
            for row in group:
                with Image.open(out/row['overlay']) as im:
                    frames.append(im.resize((864,round(im.height*864/im.width))).convert('P',palette=Image.Palette.ADAPTIVE))
            filename=f'previews/{group[0]["decode_index"]:06d}_{group[-1]["decode_index"]:06d}.gif'
            frames[0].save(out/filename,save_all=True,append_images=frames[1:],duration=400,loop=0,disposal=2)
            animations.append(dict(path=filename,indices=[r['decode_index'] for r in group],display_frame_ms=400,
                temporal_identity_verified=False,source_video_speed=False))
        unchanged();check()
        _need(all((out/'default_masks'/f'{i:06d}.png').read_bytes()==blob for i,blob in safe.items()),'default masks mutated')
        result.update(status='NARROW_GAP_CANDIDATES_NOT_ACCEPTED',frames=rows,animations=animations,source_unchanged=True,
            source_bindings=refs,policy=plan['policy'],source_grid=[plan['width'],plan['height']],
            temporal_status='NEIGHBORING_FRAME_DIAGNOSTICS_NOT_TRACK_IDENTITY',
            summary=dict(frames=len(rows),blocked_union_additions=sum(r['blocked_union_additions'] for r in rows),
                frames_with_conflict=sum(not r['before_guard']['preserved'] for r in rows),
                candidate_guard_failures=sum(not r['after_guard']['preserved'] for r in rows),
                fallback_frames=sum(r['fallback_reason'] is not None for r in rows),
                refused_frames=sum(r['status']=='REFUSED_CHANGE_CAP' for r in rows),base_foreground_lost=0,added_foreground_vs_union=0))
    except Exception as exc:
        result.update(status='FAILED',error=type(exc).__name__+': '+str(exc),frames=rows,source_unchanged=False)
    result['elapsed_seconds']=time.monotonic()-start;_save(out/'RESULT.json',result);render(out,result)
    return result


def render(out,result):
    esc=html.escape
    page='<!doctype html><meta charset="utf-8"><title>窄缝自动定位与保护</title><style>body{font:17px/1.6 system-ui;max-width:1300px;margin:35px auto;padding:0 20px;color:#172438;background:#f3f6fa}img{max-width:100%}td,th{padding:8px 12px;border-bottom:1px solid #ccd}a{color:#0965b5}.box{background:#fff5d7;padding:15px}</style><h1>窄缝自动定位与保护</h1><p>'+esc(result['status'])+'</p>'
    page+='<div class="box">这是原有背景通道的保护候选，并非已证明精准的分割。默认仍使用未改的原并集；未运行新模型或三维重建。窄缝来自每帧重新定位，不固定坐标，不假定不同帧中的缝是同一条。</div>'
    page+='<p>绿色：外连窄背景；橙色：过细、不能提供几何负证据；紫色：封闭背景；洋红：原并集填入保护区域或封闭其通路；黄色：阻止的新增前景。</p><pre>'+esc(str(result.get('summary',result.get('error',''))))+'</pre>'
    page+='<p>宽度是四个扫描方向中最小的两岸前景像素中心跨度上界，不是真实三维宽度，也不是精确最窄距离。现有掩膜网格 '+esc(str(result.get('source_grid','UNKNOWN')))+'；没有声称恢复原视频的细节。</p>'
    page+='<p><a href="RESULT.json">完整证据</a> · <a href="PLAN.json">冻结策略与来源</a></p>'
    page+='<table><tr><th>帧</th><th>被阻止的并集新增像素</th><th>并集是否破坏保护</th><th>候选保持</th><th>策略</th></tr>'
    for row in result.get('frames',[]):
        page+='<tr><td>'+str(row['decode_index'])+'</td><td>'+str(row['blocked_union_additions'])+'</td><td>'+str(not row['before_guard']['preserved'])+'</td><td>'+str(row['after_guard']['preserved'])+'</td><td>'+esc(row['fallback_reason'] or '局部阻止')+'</td></tr>'
    page+='</table><h2>邻帧慢放（非原速、非已验证轨迹）</h2>'
    for a in result.get('animations',[]):page+='<p>'+str(a['indices'])+'</p><img src="'+a['path']+'">'
    page+='<h2>逐帧全分辨率对照</h2>'
    for row in result.get('frames',[]):page+='<p>帧 '+str(row['decode_index'])+'</p><a href="'+row['overlay']+'"><img loading="lazy" src="'+row['overlay']+'"></a>'
    (Path(out)/'index.html').write_text(page,encoding='utf8')
