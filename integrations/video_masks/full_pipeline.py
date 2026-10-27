"""Produce a full-timeline mask video with explicit lineage and uncertainty."""
from __future__ import annotations

from collections import Counter
import html
import os
from pathlib import Path
import shutil
import signal
import threading
from urllib.parse import quote

from integrations.demo_pipeline.pipeline import read, save, sha, _invoke, entry_code
from .full_planning import make_full_plan
from .runner import ResourceWatch, complete_records


def automatic_boundary_complete(result: dict, required: bool = False) -> bool:
    """Keep stage success separate from a required automatic review's completion."""
    detail = result.get('automatic_boundary_review')
    required = required or (isinstance(detail, dict) and detail.get('required') is True)
    if not required:
        return True
    return (isinstance(detail, dict) and detail.get('required') is True
            and detail.get('status') == 'AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED')


def _automatic_boundary_html(run: Path, result: dict) -> str:
    detail = result.get('automatic_boundary_review')
    if not isinstance(detail, dict):
        return ''
    links = ['<a href="AUTOMATIC_BOUNDARY.json">automatic boundary evidence</a>']
    if detail.get('report'):
        target = Path(detail['report']).resolve()
        try:
            href = quote(Path(os.path.relpath(target, run.resolve())).as_posix(), safe='/')
        except ValueError:
            href = target.as_uri()
        links.append('<a href="' + html.escape(href, quote=True) + '">boundary review report</a>')
    return ('<section><h2>Automatic boundary review</h2><p>'
            + html.escape(str(detail.get('status', 'NOT_EVALUATED')))
            + ' (not segmentation or geometry acceptance)</p><p>'
            + ' · '.join(links) + '</p></section>')


def _attach_boundary_review(run: Path, result: dict, rows: list, renderer) -> dict:
    # The source RESULT and initial report already exist. Only the returned dict
    # and report receive the sidecar detail; never save this enriched dict to RESULT.
    from .auto_boundary import attach_auto_review
    reviewed = attach_auto_review(run, result)
    renderer(run, reviewed, rows)
    return reviewed


def decode_all(root: Path, run: Path, plan: dict, ffmpeg: str) -> None:
    frames=run/'frames';frames.mkdir()
    argv=[ffmpeg,'-hide_banner','-nostdin','-v','error','-threads','2','-i',plan['source_video_path'],
          '-vf',f'scale={plan["width"]}:{plan["height"]}:flags=lanczos,format=yuvj444p',
          '-fps_mode','passthrough','-start_number','0','-q:v','2','-threads','2',str(frames/'%06d.jpg')]
    entry=_invoke(root,run,'decode_full',argv,seconds=120,memory=1<<30)
    if entry_code(entry)!=0:
        raise RuntimeError('full video decode failed')
    paths=sorted(frames.glob('*.jpg'))
    if [int(p.stem) for p in paths]!=plan['expected_decode_indexes']:
        raise ValueError('decoded sequence is not the complete frozen timeline')
    from PIL import Image
    records=[]
    for i,path in enumerate(paths):
        with Image.open(path) as image:
            if image.size!=(plan['width'],plan['height']):
                raise ValueError('decoded frame dimensions differ')
        records.append(dict(decode_index=i,path=str(path),sha256=sha(path)))
    save(run/'DECODE.json',dict(frames=records,
        derivation='ffmpeg full original decode; Lanczos to canonical size; JPEG q2 4:4:4',
        same_coordinate_domain=True,byte_identical_to_original_canonical=False,
        note='Reused predictions were computed from the same source indexes and canonical coordinates with a previous JPEG derivation.'))


def prepare_tasks(run: Path, plan: dict) -> list:
    tasks=[]
    for task in plan['tasks']:
        folder=run/'windows'/task['task_id']/'frames';folder.mkdir(parents=True)
        for local,index in enumerate(range(task['start_decode_index'],task['end_decode_index']+1)):
            source=run/'frames'/f'{index:06d}.jpg'
            dest=folder/f'{local:06d}.jpg'
            # Both paths belong to this new run; no original evidence is modified.
            try:
                os.link(source,dest)
            except OSError:
                shutil.copyfile(source,dest)
        tasks.append(dict(task,frames_dir=str(folder),output_dir=str(folder.parent/'worker'),max_frames=96))
    return tasks


def _mask(item: dict, width: int, height: int):
    import numpy as np
    from PIL import Image
    path=Path(item['mask_path'])
    if sha(path)!=item['mask_sha256']:
        raise ValueError('committed mask hash drift')
    with Image.open(path) as image:
        if image.mode!='L' or image.size!=(width,height) or image.format!='PNG':
            raise ValueError('committed mask dimensions/format differ')
        pixels=np.array(image)
    if not np.isin(pixels,[0,255]).all():
        raise ValueError('committed mask is not binary')
    return pixels


def gather_predictions(run: Path, plan: dict) -> dict:
    """Read committed events even if hard termination prevented worker finalization."""
    pool={}
    def add(index,item):
        pool.setdefault(index,[]).append(item)
    for interval in plan['intervals']:
        if interval.get('reuse'):
            for frame in interval['reuse']['frames']:
                root=interval['start_decode_index'] if frame['direction']=='forward' else interval['end_decode_index']
                add(frame['decode_index'],dict(frame,root_seed_decode_index=root,chain_depth=0,
                    interval_id=interval['interval_id'],task_id=None,evidence_origin='reused_verified_run'))
    tasks={t['task_id']:t for t in plan['tasks']}
    path=run/'worker/events.jsonl'
    if not path.is_file():
        return pool
    records=complete_records(path)
    if [r['seq'] for r in records]!=list(range(len(records))):
        raise ValueError('worker event sequence is not contiguous')
    seen=set()
    for event in records:
        if event['event']!='frame_commit':
            continue
        task=tasks[event['task_id']]
        local=event['frame_id']
        if type(local) is not int or not 0<=local<task['frame_count'] or event['direction']!=task['direction']:
            raise ValueError('commit index/direction outside planned task')
        key=(task['task_id'],local)
        if key in seen:
            raise ValueError('duplicate frame commit')
        seen.add(key)
        if not Path(event['mask_path']).resolve().is_relative_to((run/'windows'/task['task_id']/'worker').resolve()):
            raise ValueError('commit outside task output')
        index=task['start_decode_index']+local
        add(index,dict(event,decode_index=index,root_seed_decode_index=task['root_seed_decode_index'],
            chain_depth=task['chain_depth'],interval_id=task['interval_id'],evidence_origin='current_fsynced_commit'))
    return pool


def assemble(run: Path, plan: dict) -> list:
    """Select existing masks; never union, dilate, interpolate pixels or hide uncertainty."""
    import numpy as np
    from .policy import choose_propagation
    pool=gather_predictions(run,plan)
    anchors={r['decode_index']:r for r in plan['original_anchors']}
    sampled={r['decode_index']:r for r in plan['coverage_ledger']}
    timeline=read(Path(plan['timeline_path']))['frames']
    rows=[]
    for index in plan['expected_decode_indexes']:
        available=pool.get(index,[])
        directions={}
        for direction in ('forward','reverse'):
            candidates=[r for r in available if r['direction']==direction]
            if candidates:
                directions[direction]=min(candidates,key=lambda r:(r['chain_depth'],
                    abs(index-r['root_seed_decode_index']),r.get('task_id') or ''))
        arrays={key:_mask(value,plan['width'],plan['height']) for key,value in directions.items()}
        independent=(len(directions)==2 and directions['forward']['root_seed_decode_index']
                     !=directions['reverse']['root_seed_decode_index'])
        decision=choose_propagation(arrays.get('forward'),arrays.get('reverse'))
        selected=None
        if index in anchors:
            selected=anchors[index];status='ORIGINAL_RETAINED'
            pixels=_mask(selected,plan['width'],plan['height'])
        elif directions:
            key=min(directions,key=lambda key:(abs(index-directions[key]['root_seed_decode_index']),
                                              directions[key]['chain_depth'],key))
            selected=directions[key];pixels=arrays[key]
            if not pixels.any():
                status='PREDICTED_EMPTY'
            elif independent:
                status='BIDIRECTIONAL_CANDIDATE' if decision['status']=='PROPAGATED_CANDIDATE' else 'DISAGREEMENT_CANDIDATE'
            else:
                status='SINGLE_DIRECTION_CANDIDATE'
        else:
            pixels=None;status='UNKNOWN'
        area=int(np.count_nonzero(pixels)) if pixels is not None else None
        centroid=None
        if area:
            ys,xs=np.where(pixels!=0);centroid=[float(xs.mean()/plan['width']),float(ys.mean()/plan['height'])]
        source=sampled.get(index)
        rows.append(dict(decode_index=index,pts=timeline[index]['pts'],time_base=timeline[index]['time_base'],
            input_index=source['input_index'] if source else None,
            original_status=source['original_status'] if source else 'NOT_SAMPLED',status=status,
            mask_path=selected['mask_path'] if selected else None,
            mask_sha256=selected['mask_sha256'] if selected else None,
            independent_bidirectional=independent,bidirectional_metrics=decision['metrics'],
            selection_rule='original retained; otherwise nearest original anchor, then lower lineage depth',
            selected_root_seed_decode_index=selected.get('root_seed_decode_index',index) if selected else None,
            selected_chain_depth=selected.get('chain_depth',0) if selected else None,
            foreground_pixels=area,centroid_normalized=centroid,warning_flags=[],
            predictions=[dict(direction=k,mask_path=v['mask_path'],mask_sha256=v['mask_sha256'],
                root_seed_decode_index=v['root_seed_decode_index'],chain_depth=v['chain_depth'],
                task_id=v.get('task_id'),evidence_origin=v['evidence_origin']) for k,v in directions.items()]))
    for previous,current in zip(rows,rows[1:]):
        if previous['foreground_pixels'] and current['foreground_pixels']:
            ratio=current['foreground_pixels']/previous['foreground_pixels']
            current['adjacent_area_ratio']=ratio
            if ratio>1.5 or ratio<2/3:
                current['warning_flags'].append('area_change_gt_1_5x_review_only')
        if previous['centroid_normalized'] and current['centroid_normalized']:
            distance=sum((a-b)**2 for a,b in zip(previous['centroid_normalized'],current['centroid_normalized']))**0.5
            current['adjacent_centroid_distance']=distance
            if distance>0.15:
                current['warning_flags'].append('centroid_change_gt_0_15_review_only')
    save(run/'ASSEMBLY.json',dict(rows=rows,source_masks_modified=False,
        counts=dict(Counter(r['status'] for r in rows)),quality_status='NOT_EVALUATED',published_for_reconstruction=False))
    return rows


def report(run: Path, result: dict, rows: list) -> None:
    counts=Counter(r['status'] for r in rows)
    risk=[r for r in rows if r['warning_flags'] or r['status'] in ('UNKNOWN','PREDICTED_EMPTY','DISAGREEMENT_CANDIDATE')]
    metric=[r for r in rows if r['bidirectional_metrics'] and r['bidirectional_metrics']['iou'] is not None]
    selected={r['decode_index']:r for r in sorted(metric,key=lambda r:r['bidirectional_metrics']['iou'])[:8]+risk[:8]}
    for i in range(0,len(rows),max(1,len(rows)//12)):
        selected[i]=rows[i]
    if rows:selected[len(rows)-1]=rows[-1]
    cards=[]
    for index,r in sorted(selected.items()):
        cards.append(f'<figure><figcaption>帧 {index} / {r["status"]} / {html.escape(str(r["warning_flags"]))}</figcaption>'
                     f'<img loading="lazy" src="export/overlay_frames/{index:06d}.jpg"></figure>')
    page='<!doctype html><meta charset="utf-8"><title>手套完整掩膜视频</title><style>body{font:16px system-ui;background:#15232d;color:#edf5fb;margin:32px}a{color:#94dcff}video{max-width:100%;height:70vh;background:#000}figure{display:inline-block;width:260px;vertical-align:top;margin:12px}img{width:100%}pre{white-space:pre-wrap}</style>'
    page+=f'<h1>完整时间轴掩膜视频</h1><p>{len(rows)} 帧，执行状态 {result["status"]}。原掩膜保留；长区间按重叠窗口传播，尾段仅单向预测。帧数完整不等于轮廓完整。</p>'
    page+='<p><a href="overlay.mp4">叠加视频</a> · <a href="masks_preview.mp4">黑白预览</a> · <a href="masks_lossless.mkv">无损掩膜视频</a> · <a href="MANIFEST.json">逐帧来源</a> · <a href="RESULT.json">执行记录</a></p>'
    if (run/'overlay.mp4').is_file():page+='<video controls preload="metadata" src="overlay.mp4"></video> <video controls preload="metadata" src="masks_preview.mp4"></video>'
    page+='<pre>'+html.escape(str(dict(counts)))+'</pre><p>绿色：具有双向证据；橙色：单向或存在分歧；青色：原始掩膜；紫色：缺失占位。双向一致也可能共同漏分。警示仅用于复核，没有自动改形、补洞或填充画外区域。</p>'
    page+='<h2>自动抽查：最小一致性、突变和均匀时间点</h2>'+''.join(cards)
    page+=_automatic_boundary_html(run,result)
    (run/'index.html').write_text(page,encoding='utf-8')


def execute(pipeline: Path, output: Path, reuse_run: Path|None=None, execute_model: bool=False,
            fusion_mode: str='none') -> dict:
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve()
    if output.exists():raise ValueError('output must be new')
    if fusion_mode not in ('none','union','supported-holes'):raise ValueError('unknown fusion mode')
    plan=make_full_plan(pipeline,reuse_run)
    plan['mask_fusion_mode']=fusion_mode
    output.mkdir(parents=True)
    save(output/'PLAN.json',plan)
    result=dict(status='PREPARED',model_entries=0,automatic_retry=False,source_masks_modified=False,
                published_for_reconstruction=False,quality_status='NOT_EVALUATED')
    save(output/'RESULT.json',result)
    if not execute_model:
        report(output,result,[])
        return _attach_boundary_review(output,result,[],report)
    inputs={plan[k+'_path']:plan[k+'_sha256'] for k in
            ('source_video','source_manifest','video_result','pipeline_plan','io_run','io_frames','timeline')}
    inputs.update(plan['reuse_inputs'])
    for row in plan['coverage_ledger']:
        inputs[row['source_image']]=row['source_image_sha256']
        if row.get('mask_path'):inputs[row['mask_path']]=row['mask_sha256']
    files=list((root/'integrations/video_masks').glob('*.py'))+list((root/'integrations/sam3/src/sam3-repo/sam3').rglob('*.py'))
    files += [root/p for p in ('scripts/run_photo_demo.py','scripts/propagate_video_masks.py',
        'integrations/demo_pipeline/pipeline.py','integrations/demo_pipeline/supervision.py',
        'integrations/sam3/sam3_adapter/observer.py','integrations/sam3/sam3_adapter/resources.py')]
    bindings={str(p):sha(p) for p in files}
    checkpoint=None;checkpoint_stat=None;rows=[]
    try:
        source_plan=read(Path(pipeline)/'PLAN.json');checkpoint=Path(source_plan['checkpoint'])
        if sha(checkpoint)!=source_plan['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        checkpoint_stat=[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]
        if any(sha(Path(p))!=digest for p,digest in inputs.items()):raise ValueError('input drift before execution')
        save(output/'START_BINDINGS.json',dict(input_files=inputs,code=bindings,checkpoint_stat=checkpoint_stat,
            checkpoint_sha256=source_plan['checkpoint_sha256']))
        result.update(status='RUNNING',limits=dict(model_wall_seconds=1200,worker_commit_bytes=12<<30,
            device_used_bytes=7<<30,job_memory_bytes=14<<30,output_bytes=3<<30))
        save(output/'RESULT.json',result)
        decode_all(root,output,plan,source_plan['video_tools']['ffmpeg'])
        tasks=prepare_tasks(output,plan)
        if tasks:
            repo=root/'integrations/sam3/src/sam3-repo'
            cfg=dict(repo_root=str(repo),checkpoint_path=str(checkpoint),
                bpe_path=str(repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),threads=2,device='cuda:0',
                output_dir=str(output/'worker'),tasks=tasks)
            save(output/'CONFIG.json',cfg)
            from integrations.demo_pipeline.supervision import supervise
            logs=output/'logs/propagate';logs.mkdir(parents=True)
            temp=output/'tmp';temp.mkdir(exist_ok=True)
            watch=ResourceWatch(output,'cuda:0');cancelled=threading.Event()
            previous=signal.signal(signal.SIGINT,lambda *_:cancelled.set())
            try:
                watch.start()
                if watch.reason:raise RuntimeError(watch.reason)
                result['model_entries']=1;save(output/'RESULT.json',result)
                entry=supervise([source_plan['sam3_python'],'-B','-X','utf8',str(root/'integrations/video_masks/full_worker.py'),
                    '--config',str(output/'CONFIG.json')],cwd=root,
                    env_overrides={**{k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','TEMP':str(temp),'TMP':str(temp)},
                    log_path=logs/'supervision.jsonl',stdout_path=logs/'stdout.txt',stderr_path=logs/'stderr.txt',
                    wall_clock_seconds=1200,job_memory_limit_bytes=14<<30,run_dir=output,run_dir_threshold_bytes=3<<30,
                    total_roots=[output],total_threshold_bytes=3<<30,poll_seconds=0.5,reclamation_seconds=15,
                    label='full-video-masks',script_sha256=sha(Path(__file__)),
                    cancel_requested=lambda:cancelled.is_set() or watch.reason is not None)
                save(logs/'ENTRY.json',entry)
                result.update(worker_exit_code=entry_code(entry),reclaim_confirmed=entry.get('reclaim_confirmed'))
            finally:
                result['monitor']=watch.close();signal.signal(signal.SIGINT,previous)
            if result['worker_exit_code']!=0 or watch.reason or not result['monitor']['joined']:
                raise RuntimeError('bounded propagation did not complete; inspect worker and supervisor evidence')
        rows=assemble(output,plan)
        if fusion_mode!='none':
            save(output/'SOURCE_ASSEMBLY.json',dict(rows=rows))
            from .fusion import combine_rows
            rows,result['fusion']=combine_rows(output,plan,rows,gather_predictions(output,plan),fusion_mode)
        save(output/'RESULT.json',dict(result,status='EXPORTING'))
        from .video_export import export_video
        result['export']=export_video(root,output,plan,rows,source_plan['video_tools']['ffmpeg'],source_plan['video_tools']['ffprobe'])
        result.update(status='FULL_TIMELINE_EXPORTED' if not result['export']['placeholder_count'] else 'PARTIAL_VIDEO_EXPORTED',
            frame_count=plan['frame_count'],counts=dict(Counter(r['status'] for r in rows)),
            warning_frames=sum(bool(r['warning_flags']) for r in rows))
    except Exception as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        if (output/'frames').is_dir() and not rows:
            try:rows=assemble(output,plan)
            except Exception as error:result['assembly_error']=str(error)
    finally:
        def changed(items):
            return [p for p,d in items.items() if not Path(p).is_file() or sha(Path(p))!=d]
        result['code_drift']=changed(bindings);result['input_drift']=changed(inputs)
        result['checkpoint_stat_unchanged']=bool(checkpoint and checkpoint_stat and checkpoint.is_file()
            and [checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]==checkpoint_stat)
        if result['code_drift'] or result['input_drift'] or not result['checkpoint_stat_unchanged']:
            result['status']='FAILED'
        save(output/'RESULT.json',result)
        if result.get('fusion'):
            from .fusion import report as fusion_report
            fusion_report(output,result,rows)
        else:
            report(output,result,rows)
    renderer=fusion_report if result.get('fusion') else report
    return _attach_boundary_review(output,result,rows,renderer)
