"""Bound automatic narrow-gap review with native RGB evidence, never mask promotion."""
from __future__ import annotations
import hashlib
import html
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from .boundary_precision import _read, _save
from . import narrow_gap_review as gap_review
from integrations.demo_pipeline.pipeline import _invoke, entry_code, LIGHT_REL

ROOT = Path(__file__).resolve().parents[2]
READY = 'AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED'


def bind(path, expected=None):
    p=Path(path).resolve(strict=True)
    if not p.is_file() or p.stat().st_size>2<<30:raise ValueError('bounded regular file required')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    r=dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
    if expected is not None and r['sha256']!=expected:raise ValueError('binding mismatch: '+str(p))
    return r


def unchanged(refs):
    drift=[r['path'] for r in refs if bind(r['path'])!=r]
    if drift:raise ValueError('input/code drift: '+str(drift))


def select_indices(rows, max_frames=16, requested=None):
    """Four temporal strata, greatest declared expansion, neighbours, then coverage."""
    if type(max_frames) is not int or not 1<=max_frames<=32:raise ValueError('1..32 frame budget')
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty fusion rows required')
    ids=[r['decode_index'] for r in rows]
    if any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):raise ValueError('unique integer indices')
    ids=sorted(ids);by={r['decode_index']:r for r in rows};scores={}
    for i in ids:
        x=by[i].get('added_fraction_of_base',0)
        if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('finite risk score required')
        scores[i]=float(x)
    if requested is not None:
        values=list(requested)
        if (not values or len(values)>max_frames or any(type(i) is not int or i not in by for i in values)
            or len(set(values))!=len(values)):raise ValueError('explicit indices exceed bound or are invalid')
        selected=sorted(values);reason='EXPLICIT_BOUND_SUBSET'
    else:
        selected=[]
        def add(i):
            if i in by and i not in selected and len(selected)<max_frames:selected.append(i)
        groups=np.array_split(np.asarray(ids),min(4,len(ids)))
        anchors=[max((int(i) for i in g),key=lambda i:(scores[i],-i)) for g in groups]
        add(ids[0]);add(ids[-1])
        for i in anchors:add(i)
        for offset in (-1,1):
            for i in anchors:add(i+offset)
        for pos in np.linspace(0,len(ids)-1,min(max_frames,len(ids))).round().astype(int):add(ids[pos])
        for i in ids:add(i)
        selected.sort();reason='FOUR_STRATA_EXPANSION_NEIGHBOURS_AND_COVERAGE'
    return dict(indices=selected,source_count=len(ids),unselected_count=len(ids)-len(selected),
        unselected_status='NOT_EVALUATED',selection_rule=reason,max_frames=max_frames,
        selected_risk=[dict(decode_index=i,added_fraction_of_base=scores[i]) for i in selected],
        risk_is_not_accuracy=True)


def decode_argv(ffmpeg, video, out, indices, width, height):
    selection='+'.join(f'eq(n\\,{i})' for i in indices)
    filters=(f'[0:v:0]select={selection},showinfo,split=2[n][c];'
        f'[n]format=rgb24[native];[c]scale={width}:{height}:flags=lanczos,format=yuvj444p[canonical]')
    return [str(ffmpeg),'-hide_banner','-nostdin','-v','info','-threads','2','-i',str(video),
        '-filter_complex_threads','2','-filter_complex',filters,
        '-map','[native]','-fps_mode','passthrough','-start_number','0','-threads','2',str(out/'native/%06d.png'),
        '-map','[canonical]','-fps_mode','passthrough','-start_number','0','-q:v','2','-threads','2',str(out/'canonical/%06d.jpg')]


def verify_decoded(out, frames, width, height, refs):
    """Byte-exact canonical re-derivation is required, not inferred from filenames."""
    native=sorted((out/'native').glob('*.png'));canonical=sorted((out/'canonical').glob('*.jpg'))
    if len(native)!=len(frames) or len(canonical)!=len(frames):raise ValueError('decoded frame count mismatch')
    result=[]
    for ordinal,frame in enumerate(frames):
        if native[ordinal].stem!=f'{ordinal:06d}' or canonical[ordinal].stem!=f'{ordinal:06d}':raise ValueError('ordinal gap')
        source=bind(frame['image']);derived=bind(canonical[ordinal])
        if source['sha256']!=derived['sha256']:
            raise ValueError('CANONICAL_REDERIVATION_MISMATCH: native/canonical association unverified; not proof of temporal drift')
        with Image.open(native[ordinal]) as im:
            if im.mode!='RGB':raise ValueError('native RGB PNG required')
            w,h=im.size
        if w<width or h<height or w*height!=h*width or w*h>4_194_304:raise ValueError('unsupported native coordinate transform')
        ref=bind(native[ordinal]);refs.extend([ref,derived])
        result.append(dict(decode_index=frame['decode_index'],ordinal=ordinal,path=str(native[ordinal]),sha256=ref['sha256'],
            size=[w,h],canonical_sha256=derived['sha256'],canonical_rederivation_byte_equal=True,
            scale=w/width,pixel_mapping='native centre=(canonical centre+0.5)*scale-0.5',
            new_segmentation=False))
    return result


def execute_auto_boundary(source, output, indices=None, max_frames=16, *, invoke=None):
    source=Path(source).resolve(strict=True);out=Path(output).resolve();invoke=invoke or _invoke
    if out.exists() or not out.is_relative_to(ROOT/'outputs') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('fresh disjoint repository output required')
    started=time.monotonic();refs=[];rows=[]
    def add(p,expected=None):
        r=bind(p,expected);refs.append(r);return Path(r['path'])
    source_plan=_read(add(source/'PLAN.json'));fusion=_read(add(source/'FUSION.json'))
    selected=select_indices(fusion['rows'],max_frames,indices)
    # prepare() validates complete same-frame union lineage before decoding.
    gap_plan=gap_review.prepare(source,out/'gaps',selected['indices'])
    video=add(source_plan['source_video_path'],source_plan['source_video_sha256'])
    pipeline_plan=_read(add(source_plan['pipeline_plan_path'],source_plan['pipeline_plan_sha256']))
    ffmpeg=add(pipeline_plan['video_tools']['ffmpeg'])
    for p in (Path(__file__),ROOT/'integrations/video_masks/native_gap_evidence.py',
        ROOT/'scripts/review_auto_boundaries.py',ROOT/'integrations/demo_pipeline/pipeline.py',
        ROOT/'integrations/demo_pipeline/supervision.py'):add(p)
    refs.extend(gap_plan['source_bindings']);refs.append(bind(out/'gaps/PLAN.json'))
    _save(out/'SELECTION.json',selected)
    _save(out/'PLAN.json',dict(schema='automatic-boundary-plan-v1',source=str(source),selection=selected,
        bindings=refs,source_video=str(video),native_decode_seconds=120,max_job_bytes=2<<30,
        default_mask_policy='UNCHANGED_UNION',promotion=False,new_model_executions=0,
        rule='Bounded diagnostic sampling. No semantic accuracy or full timeline acceptance.'))
    result=dict(status='RUNNING',source_run=str(source),source_unchanged=False,selection=selected,frames=rows,
        mask_promotion=False,new_model_executions=0,new_reconstruction_executions=0,geometry_accepted=False,
        quality_status='NOT_ACCEPTED',report=str(out/'index.html'))
    _save(out/'START.json',dict(executions=1));_save(out/'RESULT.json',result)
    try:
        unchanged(refs)
        gaps=gap_review.execute(out/'gaps')
        if gaps['status']!='NARROW_GAP_CANDIDATES_NOT_ACCEPTED':
            raise ValueError('gap candidates not verifiable')
        result['gap_summary']=gaps['summary'];refs.extend(gaps['source_bindings'])
        gap_rows={r['decode_index']:r for r in gaps['frames']}
        for d in ('native','canonical','evidence','overlays'):(out/d).mkdir()
        argv=decode_argv(ffmpeg,video,out,selected['indices'],gap_plan['width'],gap_plan['height'])
        entry=invoke(ROOT,out,'decode_native_boundaries',argv,seconds=120,memory=2<<30)
        result['native_decode_exit_code']=entry_code(entry)
        if entry_code(entry)!=0 or entry.get('reclaim_confirmed') is not True:raise RuntimeError('bounded native decode did not complete/reclaim')
        frames=verify_decoded(out,gap_plan['frames'],gap_plan['width'],gap_plan['height'],refs)
        _save(out/'DECODE_BINDINGS.json',dict(frames=frames,source_video=bind(video),
            interpretation='Bound source ordinal and exact canonical JPEG re-derivation; native RGB is not a new mask.'))
        from .native_gap_evidence import assess_native_gap
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv(refs);cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
        for f,n in zip(gap_plan['frames'],frames):
            if time.monotonic()-started>360:raise TimeoutError('automatic boundary overall CPU deadline')
            index=f['decode_index'];stem=f'{index:06d}'
            with Image.open(n['path']) as im:rgb=np.array(im)
            def mask(path):
                with Image.open(path) as im:
                    x=np.array(im)
                    if im.mode!='L' or not np.isin(x,[0,255]).all():raise ValueError('binary mask required')
                return x!=0
            base=mask(f['base']);union=mask(f['union']);proposal=mask(out/'gaps/proposals'/f'{stem}.png')
            with np.load(out/'gaps/arrays'/f'{stem}.npz',allow_pickle=False) as a:
                stored={k:a[k].copy() for k in a.files}
            # Recompute and compare: source-derived coordinates must match stored evidence.
            from .narrow_gap import detect_narrow_gaps
            diag=detect_narrow_gaps(base,np.ones_like(base),cv=cv)
            for key,value in stored.items():
                if key in diag and isinstance(diag[key],np.ndarray) and not np.array_equal(value,diag[key],equal_nan=True):
                    raise ValueError('gap coordinate evidence changed: '+key)
            measured=assess_native_gap(rgb,base,union,proposal,diag,cv=cv)
            maps=measured.pop('maps')
            np.savez_compressed(out/'evidence'/f'{stem}.npz',**maps)
            row=dict(decode_index=index,native_binding=n,**measured,overlay=f'overlays/{stem}.png',
                candidate_status=gap_rows[index]['status'],candidate_promoted=False,
                default_mask_kind='UNCHANGED_LITERAL_UNION')
            _save(out/'evidence'/f'{stem}.json',row);rows.append(row)
            # Exact evidence masks are transferred only for display; no image edits feed the model.
            canvas=Image.new('RGB',(rgb.shape[1]*2,rgb.shape[0]+56),(19,30,43));draw=ImageDraw.Draw(canvas)
            canvas.paste(Image.fromarray(rgb),(0,56));visual=rgb.copy()
            for code,color in [(1,(30,220,120)),(2,(255,70,60)),(3,(255,190,35))]:
                m=maps['evidence_code']==code
                m=cv.resize(m.astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv.INTER_NEAREST).astype(bool)
                visual[m]=color
            canvas.paste(Image.fromarray(visual),(rgb.shape[1],56))
            draw.text((8,8),f'FRAME {index} | NATIVE RGB',fill='white')
            draw.text((rgb.shape[1]+8,8),'GREEN: supported | RED: weak | AMBER: unknown',fill='white')
            draw.text((8,30),'RGB evidence only; not new segmentation or geometry acceptance',fill='white')
            canvas.save(out/row['overlay'])
            _save(out/'PROGRESS.json',dict(completed=len(rows),last_index=index))
        unchanged(refs)
        result.update(status=READY,source_unchanged=True,frames=rows,selected_frame_count=len(rows),
            full_timeline_reviewed=len(rows)==selected['source_count'])
    except Exception as exc:
        result.update(status='FAILED',error=type(exc).__name__+': '+str(exc),frames=rows)
    result.update(elapsed_seconds=time.monotonic()-started,bindings=refs)
    _save(out/'RESULT.json',result);render(out,result)
    return result


def required_union(result):
    return (result.get('status')=='FULL_TIMELINE_FUSED_EXPORTED' and result.get('mode')=='union') or (
        result.get('status')=='FULL_TIMELINE_EXPORTED' and result.get('fusion',{}).get('mode')=='union')


def attach_auto_review(run, result):
    """After frozen source result is saved: append sidecar, never rewrite RESULT."""
    run=Path(run).resolve();sidecar=run/'AUTOMATIC_BOUNDARY.json'
    detail=dict(status='NOT_APPLICABLE',required=False,report=None,result_path=None,sidecar=str(sidecar),
        reason='Completed literal-union input required',promoted=False)
    if required_union(result) and not result.get('input_drift') and not result.get('code_drift'):
        detail.update(status='NOT_EVALUATED',required=True,reason=None)
        target=run.with_name(run.name+'__boundary_review')
        try:
            if sidecar.exists() or target.exists():raise ValueError('automatic review already exists; no overwrite or retry')
            argv=[str(ROOT/LIGHT_REL),'-B','-X','utf8',str(ROOT/'scripts/review_auto_boundaries.py'),
                '--run',str(run),'--output',str(target)]
            e=_invoke(ROOT,run,'automatic_boundary',argv,seconds=420,memory=2<<30,additional_roots=[target])
            r=_read(target/'RESULT.json') if (target/'RESULT.json').is_file() else {}
            detail.update(status=r.get('status','FAILED'),report=str(target/'index.html') if (target/'index.html').is_file() else None,
                result_path=str(target/'RESULT.json'),exit_code=entry_code(e),reclaim_confirmed=e.get('reclaim_confirmed'))
            if entry_code(e)!=0 or not e.get('reclaim_confirmed') or detail['status']!=READY:
                detail.update(status='FAILED',reason=r.get('error','Automatic review failed or was not reclaimed'))
        except Exception as exc:detail.update(status='FAILED',reason=type(exc).__name__+': '+str(exc))
    # This sidecar is excluded from source PLAN/RESULT byte bindings on purpose.
    if sidecar.exists():
        # An existing audit is immutable: return a conflict without destroying it.
        return dict(result,automatic_boundary_review=dict(detail,status='FAILED',reason='Existing automatic boundary sidecar preserved'))
    _save(sidecar,detail)
    return dict(result,automatic_boundary_review=detail)


def render(out,result):
    esc=html.escape;s=result['selection']
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动边界复核</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#eef3f7;color:#173047}img{max-width:100%}pre{white-space:pre-wrap}a{color:#0671ac}.notice{background:#fff1c9;padding:16px}</style><h1>自动窄缝保护 + 原分辨率边缘证据</h1>'
    page+='<div class="notice">'+esc(result['status'])+'。候选不自动替换掩膜，不直接证明三维指缝已分开。</div>'
    page+=f'<p>全时间线 {s["source_count"]} 帧；自动选择 {len(s["indices"])} 帧：{s["indices"]}；未选 {s["unselected_count"]} 帧明确为 NOT_EVALUATED。</p>'
    page+='<p>按四个时间区间的融合扩张风险选锚帧并加入邻帧，再补覆盖帧。不会只取开头，也不固定手指位置。原图与旧帧通过同配方重建 JPEG 的字节 hash 核对；失败则停止证据判定。</p>'
    page+='<p><a href="gaps/index.html">窄缝候选与原掩膜对照</a> · <a href="SELECTION.json">自动选帧依据</a> · <a href="RESULT.json">完整记录</a></p>'
    if result.get('error'):page+='<pre>'+esc(result['error'])+'</pre>'
    for row in result.get('frames',[]):
        page+='<h2>帧 '+str(row['decode_index'])+' · 原图 '+str(row['native_binding']['size'])+'</h2><p>掩膜候选：'+esc(row['candidate_status'])+'；仍保留原并集默认值。超出改动上限的候选也展示RGB证据，但不推广。</p><pre>'+esc(str(row.get('stats',{})))+'</pre><a href="'+row['overlay']+'"><img loading="lazy" src="'+row['overlay']+'"></a>'
    page+='<p>绿：局部两岸RGB支持；红：支持弱；橙：未知。图上颜色仅为诊断叠加。阴影、纹理、遮挡都可能影响梯度，支持不等于真值；弱支持也不授权填缝。放大的旧掩膜仅用于坐标转移，不称原分辨率重新分割。</p>'
    page+='<p>下一阶段仍需跨视可见性与三角面投影检查。未运行新的SAM3、SfM/MVS或CAD。本模块是工程边界验证，非Kozlov论文的原算法。</p></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
