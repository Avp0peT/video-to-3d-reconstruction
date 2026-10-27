"""Bounded multi-interval propagation and a complete sampled-frame coverage ledger."""
from __future__ import annotations

import html
from pathlib import Path
import signal
import threading

from integrations.demo_pipeline.pipeline import read, save, sha, entry_code
from .planning import make_coverage_plan
from .runner import ResourceWatch, complete_records, decode, review


def summarize(plan: dict, reviews: dict) -> dict:
    """Keep original recognition, propagation agreement and uncovered frames separate."""
    sampled = {}
    unique = {}
    intervals = []
    for interval in plan['intervals']:
        identifier = interval['interval_id']
        result = reviews.get(identifier, {})
        rows = result.get('rows', [])
        metrics = [r['decision']['metrics']['iou'] for r in rows
                   if r['decision'].get('metrics') and r['decision']['metrics']['iou'] is not None]
        for row in rows:
            evidence = dict(interval_id=identifier, **row)
            unique.setdefault(row['decode_index'], []).append(evidence)
            if row['input_index'] is not None:
                sampled.setdefault(row['input_index'], []).append(evidence)
        worst = min((r for r in rows if r['decision'].get('metrics')
                     and r['decision']['metrics']['iou'] is not None),
                    key=lambda r: r['decision']['metrics']['iou'], default=None)
        intervals.append(dict(interval_id=identifier,
            start_decode_index=interval['start_decode_index'], end_decode_index=interval['end_decode_index'],
            frame_count=interval['frame_count'], worker_status=result.get('worker_status', 'NOT_RUN'),
            candidate_count=result.get('candidate_count', 0),
            disagreement_count=sum(r['decision']['status'] in ('UNRESOLVED','EMPTY') for r in rows),
            mean_bidirectional_iou=sum(metrics)/len(metrics) if metrics else None,
            min_bidirectional_iou=min(metrics) if metrics else None,
            worst_local_frame=worst['local_frame_index'] if worst else None,
            report=f'clips/{identifier}/index.html'))
    ledger = []
    for item in plan['coverage_ledger']:
        evidence = [r for r in sampled.get(item['input_index'], []) if r['sources']]
        if item['original_status'] == 'OK':
            status = 'ORIGINAL_RETAINED'
        elif evidence and all(r['decision']['status'] == 'PROPAGATED_CANDIDATE' for r in evidence):
            status = 'PROPAGATED_CANDIDATE'
        elif evidence:
            status = 'UNRESOLVED'
        else:
            status = 'NOT_RUN'
        ledger.append(dict(item, propagation_status=status,
                           propagation_evidence=[dict(interval_id=r['interval_id'],
                             local_frame_index=r['local_frame_index'], decision=r['decision'],
                             sources=r['sources']) for r in evidence]))
    return dict(schema='video-mask-coverage-review-v1', intervals=intervals, coverage_ledger=ledger,
        sampled_frame_count=len(ledger), original_ok_count=sum(r['original_status']=='OK' for r in ledger),
        sampled_candidate_count=sum(r['propagation_status']=='PROPAGATED_CANDIDATE' for r in ledger),
        sampled_unresolved_count=sum(r['propagation_status']=='UNRESOLVED' for r in ledger),
        sampled_not_run_count=sum(r['propagation_status']=='NOT_RUN' for r in ledger),
        unique_decode_frames_reviewed=len(unique),
        unique_frames_with_committed_mask=sum(any(r['sources'] for r in rows) for rows in unique.values()),
        unique_candidate_count=sum(all(r['decision']['status']=='PROPAGATED_CANDIDATE' for r in rows)
                                   for rows in unique.values()),
        source_masks_modified=False, published_for_reconstruction=False,
        quality_status='NOT_EVALUATED', cross_view_status='NOT_EVALUATED',
        interpretation='Bidirectional agreement is not semantic accuracy; seed masks are predictions.')


def report(output: Path, plan: dict, result: dict, reviews: dict) -> dict:
    summary = summarize(plan, reviews)
    save(output/'COVERAGE.json', summary)
    cards = []
    for row in summary['intervals']:
        identifier = row['interval_id']
        worst = row['worst_local_frame']
        preview = f'<img loading="lazy" src="clips/{identifier}/review/{worst:06d}.jpg">' if worst is not None else ''
        metric = f"{row['min_bidirectional_iou']:.3f}" if row['min_bidirectional_iou'] is not None else '未测'
        cards.append(f'<section><h2><a href="{row["report"]}">{identifier}：原视频帧 '
                     f'{row["start_decode_index"]}–{row["end_decode_index"]}</a></h2>'
                     f'<p>{row["frame_count"]} 帧；新增候选 {row["candidate_count"]}；'
                     f'未一致 {row["disagreement_count"]}；最低双向 IoU {metric}。</p>{preview}</section>')
    rows = []
    for item in summary['coverage_ledger']:
        links = []
        for e in item['propagation_evidence']:
            links.append(f'<a href="clips/{e["interval_id"]}/review/{e["local_frame_index"]:06d}.jpg">对比图</a>')
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in
            (item['input_index'],item['decode_index'],item['original_status'],item['propagation_status'],item['reason'] or ''))
                    +'<td>'+' · '.join(links)+'</td></tr>')
    page = '<!doctype html><meta charset="utf-8"><title>多区间视频掩膜复核</title><style>body{background:#13212a;color:#edf4f9;font:16px system-ui;margin:32px;max-width:1200px}a{color:#89d7ff}img{max-width:100%}section{padding:16px;background:#203340;margin:24px 0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #456;padding:8px;text-align:left}</style>'
    page += '<h1>手套视频：多角度连续掩膜传播</h1><p>自动选择原始 OK 掩膜之间的缺口；所有区间共享一次模型加载，各区间与方向独立初始化。原掩膜与漏检记录保留。</p>'
    page += f'<p>执行状态：{html.escape(result["status"])}。原采样 {summary["sampled_frame_count"]} 张：原 OK {summary["original_ok_count"]}，新增一致候选 {summary["sampled_candidate_count"]}，传播仍有分歧 {summary["sampled_unresolved_count"]}，未覆盖 {summary["sampled_not_run_count"]}。</p>'
    page += '<p>候选仅供复核，尚未送入建模。端点种子也可能漏分；双向一致不能证明手指、轮廓或背景准确。下图显示每个区间双向一致性最差的一帧，点击标题查看完整动画与所有帧。</p><p><a href="PLAN.json">冻结计划</a> · <a href="COVERAGE.json">完整覆盖账本</a> · <a href="RESULT.json">执行证据</a></p>'
    if result.get('error'):
        page += '<p>'+html.escape(result['error'])+'</p>'
    page += ''.join(cards)+f'<h2>原始 {summary["sampled_frame_count"]} 张采样的去向</h2><table><tr><th>采样</th><th>视频帧</th><th>原状态</th><th>本次结果</th><th>未覆盖原因</th><th>证据</th></tr>'+''.join(rows)+'</table>'
    (output/'index.html').write_text(page,encoding='utf-8')
    return summary


def recover_commits(output: Path, interval: dict) -> None:
    """Recover only fsynced frame commits when a killed worker could not finalize."""
    folder=output/'clips'/interval['interval_id']/'worker'
    if (folder/'result.json').is_file():
        return
    event_path=output/'worker/events.jsonl'
    if not event_path.is_file():
        return
    records=complete_records(event_path)
    sequence=[r['seq'] for r in records]
    if sequence != list(range(len(sequence))):
        raise ValueError('batch event sequence is not contiguous')
    commits=[]
    seen=set()
    for record in records:
        if record.get('clip_id')!=interval['interval_id'] or record.get('event')!='frame_commit':
            continue
        key=(record['direction'],record['frame_id'])
        if key in seen or key[0] not in ('forward','reverse') or type(key[1]) is not int or not 0<=key[1]<interval['frame_count']:
            raise ValueError('invalid or duplicate batch frame commit')
        seen.add(key)
        commits.append(record)
    if commits:
        # review() checks scope, bytes, dimensions and binary labels before use.
        save(folder/'result.json',dict(status='INTERRUPTED_WITH_COMMITS',frames=commits,
            record_origin='parent_recovery_from_fsynced_events',source_events=str(event_path),
            source_events_sha256=sha(event_path),frame_count=interval['frame_count']))


def execute(pipeline: Path, output: Path, max_frames: int = 96,
            execute_model: bool = False, max_intervals: int = 6, max_total_frames: int = 384) -> dict:
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('output must be new; previous attempts are immutable')
    try:
        return _execute(pipeline,output,max_frames,execute_model,max_intervals,max_total_frames)
    except Exception as exc:
        if not output.exists():
            raise
        result=read(output/'RESULT.json') if (output/'RESULT.json').is_file() else {'model_entries':0}
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',published_for_reconstruction=False)
        save(output/'RESULT.json',result)
        try:
            report(output,read(output/'PLAN.json'),result,{})
        except Exception as report_error:
            result['report_error']=str(report_error)
            save(output/'RESULT.json',result)
        return result


def _execute(pipeline: Path, output: Path, max_frames: int,
             execute_model: bool, max_intervals: int, max_total_frames: int) -> dict:
    root = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output must be new; previous attempts are immutable')
    plan = make_coverage_plan(pipeline,max_frames,max_intervals,max_total_frames)
    output.mkdir(parents=True)
    save(output/'PLAN.json',plan)
    result = dict(status='PREPARED' if plan['intervals'] else 'NOT_APPLICABLE', model_entries=0,
                  automatic_retry=False, source_masks_modified=False, published_for_reconstruction=False)
    save(output/'RESULT.json',result)
    if not execute_model or not plan['intervals']:
        result['summary'] = report(output,plan,result,{})
        save(output/'RESULT.json',result)
        return result
    source_plan = read(Path(pipeline)/'PLAN.json')
    checkpoint = Path(source_plan['checkpoint'])
    if sha(checkpoint) != source_plan['checkpoint_sha256']:
        raise ValueError('checkpoint binding changed')
    inputs = {plan[name+'_path']:plan[name+'_sha256'] for name in
              ('source_video','source_manifest','video_result','pipeline_plan','io_run','io_frames','timeline')}
    for item in plan['coverage_ledger']:
        inputs[item['source_image']] = item['source_image_sha256']
        if item.get('mask_path'):
            inputs[item['mask_path']] = item['mask_sha256']
    for path,digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError('planned input drift: '+path)
    stat = [checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]
    code_files = (list((root/'integrations/video_masks').glob('*.py'))
        +list((root/'integrations/sam3/src/sam3-repo/sam3').rglob('*.py'))
        +[root/p for p in ('scripts/run_photo_demo.py','scripts/propagate_video_masks.py',
          'integrations/demo_pipeline/pipeline.py','integrations/demo_pipeline/supervision.py',
          'integrations/sam3/sam3_adapter/observer.py','integrations/sam3/sam3_adapter/resources.py')])
    code = {str(p):sha(p) for p in code_files}
    save(output/'START_BINDINGS.json',dict(input_files=inputs,code=code,checkpoint_stat=stat,
                                         checkpoint_sha256=source_plan['checkpoint_sha256']))
    result.update(status='RUNNING',limits=dict(worker_commit_bytes=12<<30,device_used_bytes=7<<30,
                  job_commit_bytes=14<<30,model_wall_seconds=600,output_bytes=1<<30))
    save(output/'RESULT.json',result)
    reviews = {}
    try:
        clips = []
        for interval in plan['intervals']:
            folder = output/'clips'/interval['interval_id']; folder.mkdir(parents=True)
            save(folder/'PLAN.json',interval)
            decode(root,folder,interval,source_plan['video_tools']['ffmpeg'])
            clips.append(dict(clip_id=interval['interval_id'],frames_dir=str(folder/'frames'),
                output_dir=str(folder/'worker'),frame_count=interval['frame_count'],max_frames=max_frames,
                direction='both',seeds=[dict(frame_index=s['local_frame_index'],mask_path=s['mask_path'],
                            mask_sha256=s['mask_sha256']) for s in interval['seeds']]))
        repo = root/'integrations/sam3/src/sam3-repo'
        cfg = dict(repo_root=str(repo),checkpoint_path=str(checkpoint),
            bpe_path=str(repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),device='cuda:0',threads=2,
            output_dir=str(output/'worker'),clips=clips)
        save(output/'CONFIG.json',cfg)
        from integrations.demo_pipeline.supervision import supervise
        watch = ResourceWatch(output,cfg['device'])
        logs=output/'logs/propagate'; logs.mkdir(parents=True)
        temp=output/'tmp'; temp.mkdir()
        cancelled=threading.Event()
        previous=signal.signal(signal.SIGINT,lambda *_:cancelled.set())
        try:
            watch.start()
            if watch.reason:
                raise RuntimeError(watch.reason)
            result['model_entries']=1
            save(output/'RESULT.json',result)
            entry=supervise([source_plan['sam3_python'],'-B','-X','utf8',
                str(root/'integrations/video_masks/batch_worker.py'),'--config',str(output/'CONFIG.json')],
                cwd=root,env_overrides={**{k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                    'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','TEMP':str(temp),'TMP':str(temp)},
                log_path=logs/'supervision.jsonl',stdout_path=logs/'stdout.txt',stderr_path=logs/'stderr.txt',
                wall_clock_seconds=600,job_memory_limit_bytes=14<<30,run_dir=output,
                run_dir_threshold_bytes=1<<30,total_roots=[output],total_threshold_bytes=1<<30,
                poll_seconds=0.5,reclamation_seconds=15,label='temporal-mask-batch',script_sha256=sha(Path(__file__)),
                cancel_requested=lambda:cancelled.is_set() or watch.reason is not None)
            save(logs/'ENTRY.json',entry)
            result.update(exit_code=entry_code(entry),reclaim_confirmed=entry.get('reclaim_confirmed'))
        finally:
            result['monitor']=watch.close()
            signal.signal(signal.SIGINT,previous)
        result['status']='REVIEW_READY' if (result['exit_code']==0 and not watch.reason
                             and result['monitor']['joined'] and result['reclaim_confirmed']) else 'FAILED'
    except Exception as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        for interval in plan['intervals']:
            folder=output/'clips'/interval['interval_id']
            if not (folder/'frames').is_dir():
                continue
            try:
                recover_commits(output,interval)
                reviews[interval['interval_id']]=review(folder,interval)
                save(folder/'RESULT.json',dict(status=reviews[interval['interval_id']]['worker_status'],
                                               batch_result='../../RESULT.json',published_for_reconstruction=False))
            except Exception as exc:
                result.setdefault('review_errors',[]).append(dict(interval_id=interval['interval_id'],error=str(exc)))
                result['status']='FAILED'
        result['code_drift']=[p for p,d in code.items() if not Path(p).is_file() or sha(Path(p))!=d]
        result['input_drift']=[p for p,d in inputs.items() if not Path(p).is_file() or sha(Path(p))!=d]
        result['checkpoint_stat_unchanged']=checkpoint.is_file() and [checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]==stat
        if result['code_drift'] or result['input_drift'] or not result['checkpoint_stat_unchanged']:
            result['status']='FAILED'
        result['summary']=report(output,plan,result,reviews)
        save(output/'RESULT.json',result)
    return result
