"""New, isolated candidate/geometry run from a bound existing video pipeline.

No SAM execution, alteration of old features or invented missing masks. The two
arms share decoded RGB, sampled masks, LK acceptance and native geometry policy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('sift_control', 'balanced')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def source_inputs(pipeline: Path) -> dict:
    p = read(pipeline/'pipeline.json')
    vr = pipeline/'input/video'; video = read(vr/'RESULT.json'); timeline = read(vr/'timeline.json')
    if video.get('status') != 'SUCCESS' or not 2 <= len(timeline['frames']) <= 2000:
        raise ValueError('successful bounded video pipeline required')
    source = Path(video['source_video'])
    if sha(source) != video['source_sha256'] or video['source_sha256_after'] != video['source_sha256']:
        raise ValueError('video identity drift')
    mp = Path(p['stages']['segmentation']['run_dir'])/'manifest.json'; manifest = read(mp)
    io = Path(manifest['source_io']['path'])
    for name, field in [('run.json','run_json_sha256'), ('frames.jsonl','frames_jsonl_sha256')]:
        if sha(io/name) != manifest['source_io'][field]: raise ValueError('photo-io identity drift')
    sfmp = Path(p['stages']['sfm']['run_json']); sfm = read(sfmp)
    maps = {r['input_index']:r for r in rows(sfmp.parent/'frames_map.jsonl')}
    videos = {r['photo_input_index']:r for r in video['frames'] if r.get('status') == 'PHOTO_IO_OK'}
    ios = {r['input_index']:r for r in rows(io/'frames.jsonl')}
    paths = [pipeline/'pipeline.json', vr/'RESULT.json', vr/'timeline.json', vr/'extract.process.json',
             source, mp, io/'run.json', io/'frames.jsonl', sfmp, sfmp.parent/'frames_map.jsonl', sfmp.parent/'database.db']
    frames = []
    for r in manifest['frames']:
        idx = r['input_index']; v = videos[idx]; f = ios[idx]
        if not (r['frame_id'] == v['photo_frame_id'] == f['frame_id']): raise ValueError('frame ID differs')
        image = Path(r['source_image']).resolve(strict=True)
        if image != (io/f['image_output']).resolve() or sha(image) != r['source_image_sha256']:
            raise ValueError('canonical image drift')
        item = dict(input_index=idx, frame_id=r['frame_id'], decode_index=v['decode_index'],
                    pts=v['pts'], time_seconds=v['time_seconds'], image_path=str(image),
                    image_sha256=r['source_image_sha256'], decoded_rgb_sha256=v['decoded_rgb_sha256'],
                    width=r['width'], height=r['height'], original_mask_status=r['status'])
        paths.append(image)
        if r['status'] == 'OK':
            selected = r['selected_mask']; mask = (mp.parent/selected['path']).resolve(strict=True)
            if not mask.is_relative_to(mp.parent.resolve()) or sha(mask) != selected['sha256']:
                raise ValueError('selected mask drift')
            m = maps[idx]
            if m['canonical_image_sha256'] != r['source_image_sha256']: raise ValueError('SIFT input differs')
            item.update(mask_path=str(mask), mask_sha256=selected['sha256'],
                        database_image_id=m['database_image_id'], staged_image_name=m['staged_image_name'])
            paths.append(mask)
        frames.append(item)
    frames.sort(key=lambda f:f['decode_index'])
    if len({f['decode_index'] for f in frames}) != len(frames): raise ValueError('duplicate decode frame')
    extract = read(vr/'extract.process.json'); tool = Path(extract['argv'][0])
    if sha(tool) != video['tools'][str(tool)]: raise ValueError('decoder tool drift')
    paths.append(tool)
    return dict(pipeline=str(pipeline), video_root=str(vr), source_video=str(source), frames=frames,
                decoded_count=len(timeline['frames']), database=str(sfmp.parent/'database.db'),
                baseline_sfm=str(sfmp), baseline_models=sfm['models'], extract_argv=extract['argv'],
                bindings=[binding(q) for q in sorted(set(paths))],
                sar_caveat='Existing source SAR declaration remains an assumption, not camera calibration.')


def check_bindings(items):
    return [r['path'] for r in items if not Path(r['path']).is_file() or binding(r['path']) != r]


def decode_all(inputs: dict, output: Path) -> Path:
    from scripts.prepare_video import bounded_run, inspect_image
    from .intermediate_decode import _recipe, SUFFIX
    from integrations.video_io.timeline import parse_showinfo, validate_export
    timeline = read(Path(inputs['video_root'])/'timeline.json')
    argv = list(inputs['extract_argv'])
    vi = _recipe(argv, Path(inputs['source_video']), timeline['stream_index'])
    images = output/'decoded'; images.mkdir(); logs = output/'decode_logs'; logs.mkdir()
    argv[vi] = "select='1'" + SUFFIX; argv[-1] = str(images/'frame_%06d.png')
    _, err = bounded_run(argv, logs, 'extract', time.monotonic()+120, images,
                         {'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'})
    shown = parse_showinfo(err.read_text(encoding='utf-8',errors='replace'))
    validate_export(timeline,timeline['frames'],shown)
    if len(list(images.glob('frame_*.png'))) != inputs['decoded_count']: raise ValueError('decode count differs')
    checks = []
    for f in inputs['frames']:
        info = inspect_image(images/f"frame_{f['decode_index']:06d}.png")
        if info['decoded_rgb_sha256'] != f['decoded_rgb_sha256'] or (info['width'],info['height']) != (f['width'],f['height']):
            raise ValueError('redecoded canonical RGB differs')
        checks.append(dict(input_index=f['input_index'],decode_index=f['decode_index'],rgb_equal=True))
    save(output/'DECODE.json',dict(frames=inputs['decoded_count'],sampled_rgb_checks=checks,
         intermediate_mask_status='UNKNOWN_NOT_SEGMENTED',new_segmentation_calls=0))
    return images


def track(inputs: dict, output: Path) -> dict:
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .balanced_object_tracks import Profile, seed_balanced, step_tracks, coverage, eroded_mask, points_in_mask, rebalance
    profile = Profile(); refs=[]; cv = load_cv(refs)
    if cv is None: raise RuntimeError('existing OpenCV toolchain missing')
    cv.setNumThreads(2); save(output/'CV_BINDINGS.json',refs)
    images = decode_all(inputs,output); started=time.monotonic()
    sampled = {f['decode_index']:f for f in inputs['frames']}
    arms = {name:dict(xy=np.empty((0,2),np.float32), ids=np.empty(0,np.int64), next_id=0, frames=[], stats=[], failures=Counter()) for name in ARMS}
    db = sqlite3.connect(Path(inputs['database']).as_uri()+'?mode=ro',uri=True)
    previous=None; histories=output/'steps.jsonl'
    with histories.open('x',encoding='utf-8') as log:
        for di in range(inputs['decoded_count']):
            if time.monotonic()-started>360: raise TimeoutError('tracking 360s exceeded')
            path=images/f'frame_{di:06d}.png'
            with Image.open(path) as im: gray=cv.cvtColor(np.array(im.convert('RGB')),cv.COLOR_RGB2GRAY)
            f=sampled.get(di)
            mask=None
            if f and f['original_mask_status']=='OK':
                with Image.open(f['mask_path']) as im: mask=np.asarray(im).copy()
                if mask.shape != gray.shape: raise ValueError('mask shape drift')
                safe=eroded_mask(mask,cv,profile)
            for name,a in arms.items():
                step=dict(decode_index=di,arm=name,started=len(a['ids']),mask_status=f['original_mask_status'] if f else 'UNKNOWN_NOT_SEGMENTED')
                if previous is not None and len(a['ids']):
                    move=step_tracks(previous,gray,a['xy'],cv,profile); accepted=move['accepted']
                    reasons=Counter(str(r) for r in move['reasons'][~accepted]); a['failures'].update(reasons)
                    step['rejections']=dict(reasons)
                    a['xy']=move['xy'][accepted]; a['ids']=a['ids'][accepted]
                continuing=len(a['ids']); born=np.empty((0,2),np.float32)
                if f and mask is None:
                    step['ended_unknown_mask']=len(a['ids'])
                    a['failures']['sample_without_committed_mask']+=len(a['ids'])
                    a['xy']=np.empty((0,2),np.float32); a['ids']=np.empty(0,np.int64)
                elif mask is not None:
                    inside=points_in_mask(safe,a['xy']); step['mask_rejected']=int((~inside).sum())
                    a['failures']['outside_sample_mask']+=int((~inside).sum())
                    a['xy']=a['xy'][inside]; a['ids']=a['ids'][inside]; continuing=len(a['ids'])
                    if name=='balanced':
                        keep=rebalance(mask,a['xy'],a['ids'],profile)
                        step['quota_retired']=int((~keep).sum());a['failures']['quota_retired']+=int((~keep).sum())
                        a['xy']=a['xy'][keep];a['ids']=a['ids'][keep];continuing=len(a['ids'])
                        born,_=seed_balanced(gray,mask,a['xy'],cv,profile)
                    else:
                        n,cols,payload=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(f['database_image_id'],)).fetchone()
                        candidates=np.frombuffer(payload or b'',np.float32).reshape(n,cols)[:,:2]-.5
                        if db.execute('SELECT name FROM images WHERE image_id=?',(f['database_image_id'],)).fetchone()[0] != f['staged_image_name']:
                            raise ValueError('SIFT source ID drift')
                        candidates=candidates[points_in_mask(safe,candidates)]; chosen=[]
                        for q in candidates:
                            nearby=np.concatenate((a['xy'],np.asarray(chosen,np.float32).reshape(-1,2)))
                            if len(nearby) and np.any(np.sum((nearby-q)**2,axis=1)<profile.min_distance**2):continue
                            chosen.append(q)
                            if len(nearby)+1>=profile.max_live:break
                        born=np.asarray(chosen,np.float32).reshape(-1,2) if len(a['ids'])<profile.max_live else born
                    count=len(born); new_ids=np.arange(a['next_id'],a['next_id']+count,dtype=np.int64); a['next_id']+=count
                    a['xy']=np.concatenate((a['xy'],born)); a['ids']=np.concatenate((a['ids'],new_ids))
                    obs=[dict(track_id=int(tid),xy=xy.tolist()) for tid,xy in zip(a['ids'],a['xy'])]
                    a['frames'].append({**f,'observations':obs})
                    a['stats'].append(dict(input_index=f['input_index'],decode_index=di,continued=continuing,born=count,
                        candidates=len(obs),coverage=coverage(mask,a['xy']),continued_coverage=coverage(mask,a['xy'][:continuing])))
                step.update(kept=len(a['ids']),new_seeds=len(born)); log.write(json.dumps(step,allow_nan=False)+'\n'); log.flush()
            previous=gray
            if f: print(f"sample {f['input_index']}: "+', '.join(f'{k}={len(v["ids"])}' for k,v in arms.items()),flush=True)
    db.close()
    summary={}
    for name,a in arms.items():
        dest=output/name;dest.mkdir()
        freq=Counter(o['track_id'] for f in a['frames'] for o in f['observations'])
        for f,s in zip(a['frames'],a['stats']):
            with Image.open(f['mask_path']) as im:mask=np.asarray(im)
            supported=[o['xy'] for o in f['observations'] if freq[o['track_id']]>=3]
            s.update(three_sample_candidates=len(supported),three_sample_coverage=coverage(mask,supported))
        doc=dict(schema='balanced-object-observations-v1',arm=name,frames=a['frames'],profile=asdict(profile),
             coordinates='OpenCV pixel center 0; native adapter adds 0.5 exactly once',
             new_masks=0,mask_gap_policy='terminate at sampled NO_MATCH; intermediate masks unknown',
             statistics=a['stats'],failures=dict(a['failures']),source_bindings=inputs['bindings'],
             tracking_identity='candidate temporal identity; not independent physical ground truth')
        save(dest/'TRACKS.json',doc)
        summary[name]=dict(births=a['next_id'],sampled_frames=len(a['frames']),observations=sum(len(f['observations']) for f in a['frames']),
             tracks_with_three_samples=sum(n>=3 for n in freq.values()),statistics=a['stats'],failures=dict(a['failures']))
    drift=check_bindings(inputs['bindings'])
    if drift: raise ValueError('input drift: '+str(drift))
    result=dict(status='TRACKING_COMPLETE',seconds=time.monotonic()-started,decoded_frames=inputs['decoded_count'],arms=summary,source_unchanged=True)
    save(output/'TRACKING_RESULT.json',result);return result


def launch(output: Path, name: str, args: list[str], seconds: int) -> dict:
    from .supervision import supervise
    directory=output/'processes'/name;directory.mkdir(parents=True)
    tmp=directory/'tmp';tmp.mkdir()
    command=[sys.executable,'-B','-X','utf8',str(ROOT/'scripts/optimize_object_points.py'),*args]
    env={k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(tmp),TEMP=str(tmp))
    result=supervise(command,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
         stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',wall_clock_seconds=seconds,
         job_memory_limit_bytes=8<<30,run_dir=output,run_dir_threshold_bytes=2<<30,poll_seconds=.5,
         reclamation_seconds=15,label=name,script_sha256=sha(ROOT/'scripts/optimize_object_points.py'))
    save(directory/'ENTRY.json',result)
    return result


def execute(pipeline: Path, output: Path) -> int:
    if output.exists():raise ValueError('fresh output directory required')
    if output.is_relative_to(pipeline):raise ValueError('output cannot modify the old pipeline tree')
    inputs=source_inputs(pipeline)
    output.mkdir(parents=True);save(output/'INPUTS.json',inputs)
    from .balanced_object_tracks import Profile
    implementation=[binding(ROOT/p) for p in ('scripts/optimize_object_points.py','integrations/demo_pipeline/balanced_point_run.py',
         'integrations/demo_pipeline/balanced_object_tracks.py','integrations/demo_pipeline/balanced_native.py',
         'integrations/demo_pipeline/balanced_point_report.py')]
    save(output/'PLAN.json',dict(profile=asdict(Profile()),arms=list(ARMS),implementation=implementation,
         decode_every_original_frame=True,unknown_sample_breaks_tracks=True,matching='continuous LK candidate IDs then native verification',
         minimum_samples_for_native=3,no_original_modifications=True,segmentation_calls=0,retries=0,
         budget=dict(tracking_with_decode_seconds=510,each_native_seconds=240,job_commit_bytes=8<<30,output_bytes=2<<30)))
    result=dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(output/'RESULT.json',result)
    try:
        entry=launch(output,'tracking',['track','--output',str(output)],510);result['stages']['tracking']=entry
        if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'):raise RuntimeError('tracking failed; inspect process logs')
        for arm in ARMS:
            entry=launch(output,'native_'+arm,['native','--output',str(output),'--arm',arm],240)
            result['stages']['native_'+arm]=entry
            save(output/'RESULT.json',result)
        result['status']='COMPARISON_COMPLETE' if all(v.get('root_exit_code')==0 and v.get('reclaim_confirmed') for v in result['stages'].values()) else 'PARTIAL'
    except Exception as e:
        result.update(status='FAILED',error=type(e).__name__+': '+str(e))
    finally:
        result['input_drift']=check_bindings(inputs['bindings']); result['implementation_drift']=check_bindings(implementation)
        if result['input_drift'] or result['implementation_drift']:result['status']='FAILED'
        save(output/'RESULT.json',result)
    if (output/'TRACKING_RESULT.json').is_file():
        try:
            from .balanced_point_report import build
            build(output)
        except Exception as error:
            save(output/'REPORT_FAILURE.json',dict(error=type(error).__name__+': '+str(error)))
            return 1
    return 0 if result['status']=='COMPARISON_COMPLETE' else 1


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','track','native','report'])
    parser.add_argument('--pipeline',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args(argv);output=args.output.resolve()
    if args.action=='run':
        if args.pipeline is None:parser.error('--pipeline required for run')
        return execute(args.pipeline.resolve(strict=True),output)
    if args.action=='track':
        track(read(output/'INPUTS.json'),output);return 0
    if args.action=='report':
        from .balanced_point_report import build
        build(output);return 0
    if args.arm is None:parser.error('--arm required for native')
    from .balanced_native import run
    result=run(output/args.arm/'TRACKS.json',output/args.arm/'native')
    print(json.dumps({k:result[k] for k in ('status','arm','input_count','unique_registered_count')},ensure_ascii=False,allow_nan=False))
    return 0
