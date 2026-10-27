"""Compose photo/video input, reconstruction, SAM3, surface and Blender export.

Each native stage runs once under the existing Windows Job supervisor. Reports
remain useful after failure, and report-only operations never start native work.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
LIGHT_REL = 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe'
HEAVY_REL = 'integrations/sam3/env/Scripts/python.exe'
PACKAGE_REL = 'outputs/delegation/DEMO_RUNTIME/20261011_text_v2'
MVS_REL = 'outputs/toolchains/openmvs-2.4.0-cpu/vc17/x64/Release'
CHECKPOINT_SHA = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def read(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict): raise ValueError('JSON root must be an object: ' + str(path))
    return value


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


@dataclass
class Options:
    source: Path
    prompt: str
    output: Path | None = None
    object_id: str = 'object-01'
    frames: int = 24
    ffmpeg: str | None = None
    ffprobe: str | None = None
    python: Path | None = None
    sam3_python: Path | None = None
    sam3_package: Path | None = None
    checkpoint: Path = Path(r'C:\sam3\sam3.pt')
    target: str = 'blender'
    model_id: str | None = None
    mvs_tool_root: Path | None = None
    blender_exe: Path | None = None
    view_geometry_audit: bool = True
    research_code6: bool = True
    post_review: bool = True
    freecad_python: Path | None = None
    git_exe: Path | None = None
    point_selection: str = 'native'
    all_frames: bool = False
    automatic_recovery: bool = True
    recovery_prompt: str | None = None


def source_info(source: Path) -> dict:
    source = source.expanduser().resolve(strict=True)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if not 2 <= len(files) <= 96: raise ValueError('照片目录必须含 2–96 张 JPEG/PNG；不扫描子目录。')
        kind = 'photos'
    elif source.is_file() and source.suffix.lower() in ('.mp4','.mov','.mkv','.avi','.m4v'):
        files = [source]; kind = 'video'
        if source.stat().st_size > 512 << 20: raise ValueError('视频超过现有 512 MiB 输入上限。')
    else: raise ValueError('请输入照片文件夹或 MP4/MOV/MKV/AVI 视频。')
    return {'path': str(source), 'kind': kind, 'files': [{'path': str(p), 'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}


def plan(options: Options, root: Path = ROOT) -> dict:
    if options.point_selection not in ('native','rcover-v1'): raise ValueError('point_selection must be native or rcover-v1')
    if not options.prompt.strip() or len(options.prompt) > 500: raise ValueError('请填写目标物体的简短分割提示词。')
    if not options.object_id.strip() or len(options.object_id) > 120: raise ValueError('invalid object_id')
    if type(options.frames) is not int or not 2 <= options.frames <= 96: raise ValueError('video frames must be 2–96')
    if type(options.all_frames) is not bool: raise ValueError('all_frames must be boolean')
    if type(options.automatic_recovery) is not bool: raise ValueError('automatic_recovery must be boolean')
    if options.recovery_prompt is not None and (not isinstance(options.recovery_prompt,str) or not options.recovery_prompt.strip() or len(options.recovery_prompt)>500):
        raise ValueError('recovery_prompt must be a nonempty description up to 500 characters')
    if type(options.view_geometry_audit) is not bool: raise ValueError('view_geometry_audit must be boolean')
    if options.target not in ('blender','report','cad'): raise ValueError('target must be blender, report or cad')
    if options.model_id is not None and (not isinstance(options.model_id,str) or not options.model_id.strip()):
        raise ValueError('model_id must be a nonempty string')
    info = source_info(options.source)
    if options.all_frames and info['kind']!='video':raise ValueError('all_frames requires a video source')
    from integrations.demo_pipeline.runtime import discover_git
    git_exe=discover_git(options.git_exe)
    light = (options.python or root/LIGHT_REL).resolve()
    heavy = (options.sam3_python or root/HEAVY_REL).resolve()
    package = (options.sam3_package or root/PACKAGE_REL).resolve()
    mvs = (options.mvs_tool_root or root/MVS_REL).expanduser().resolve()
    from integrations.blender_export.export import find_blender
    try: blender = str(find_blender(options.blender_exe))
    except FileNotFoundError: blender = str(options.blender_exe.resolve()) if options.blender_exe else None
    tools = {name: str(Path(explicit).expanduser().resolve()) if explicit else shutil.which(name)
             for name, explicit in [('ffmpeg',options.ffmpeg),('ffprobe',options.ffprobe)]}
    dependencies = {'project_python':light.is_file(),'sam3_python':heavy.is_file(),
                    'sam3_package':(package/'BUNDLE.json').is_file(),'checkpoint':options.checkpoint.is_file(),
                    'git_exe':git_exe is not None}
    if info['kind']=='video': dependencies.update({name:bool(path and Path(path).is_file()) for name,path in tools.items()})
    if options.target in ('blender','cad'):
        dependencies.update(openmvs=all((mvs/name).is_file() for name in ('InterfaceCOLMAP.exe','DensifyPointCloud.exe','ReconstructMesh.exe')),
                            blender=bool(blender and Path(blender).is_file()))
    freecad=(options.freecad_python or root/'outputs/toolchains/freecad-1.1.4/FreeCAD_1.1.4-Windows-x86_64-py311/bin/python.exe').expanduser().resolve()
    if options.target=='cad':dependencies['freecad_python']=freecad.is_file()
    return {'schema':'photo-demo-plan-v1','source':info,'prompt':options.prompt,'object_id':options.object_id,
            'python':str(light),'sam3_python':str(heavy),'sam3_package':str(package),'git_exe':str(git_exe) if git_exe else None,
            'checkpoint':str(options.checkpoint.resolve()),'checkpoint_sha256':CHECKPOINT_SHA,
            'video_frames':None if options.all_frames else options.frames,
            'all_frames':options.all_frames,'video_frame_limit':192 if options.all_frames else 96,
            'video_selection':'all_decoded_frames' if options.all_frames else 'uniform_presentation_time_nearest',
            'sfm_profile':'cad-object-video-v1' if options.all_frames else 'cad-object-v1',
            'segmentation_max_batches':2 if options.all_frames else 1,
            'automatic_recovery':options.automatic_recovery,
            'recovery_prompt':options.recovery_prompt,
            'recovery_policy':{'version':'object-recovery-v1','segmentation_attempts':1,
                'segmentation_trigger':'NO_MATCH or NEEDS_SELECTION, at most 96 unresolved inputs',
                'segmentation_prompt':options.recovery_prompt or 'the '+options.prompt.strip(),
                'preserve_existing_OK':True,'camera_attempts':1,
                'camera_trigger':'selected model fails native inverse-domain check; no explicit model; at most 96 inputs',
                'camera_profile':'cad-object-pinhole-v1','camera_accuracy':'NOT_EVALUATED'},
            'video_tools':tools,'dependencies':dependencies,
            'target':options.target,'model_id':options.model_id,'mvs_tool_root':str(mvs),'blender_exe':blender,'freecad_python':str(freecad),
            'stages':['input','segmentation','sfm'] + (['view_geometry'] if options.view_geometry_audit else []) + ['viewer'] + (['surface','blender'] if options.target in ('blender','cad') else []) + (['research'] if options.research_code6 else []) + (['cad'] if options.target=='cad' else []) + (['point_review','post_review'] if options.post_review else []) + ['report'],
            'post_review':options.post_review,'research_code6':options.research_code6,'view_geometry_audit':options.view_geometry_audit,
            'budgets':{'input_seconds':180,'sfm_seconds':720,'segmentation_seconds':950,
                       'point_review_seconds':180,'point_review_commit_bytes':2<<30,
                       'view_geometry_seconds':90,'view_geometry_commit_bytes':1<<30,'threads':2,'sam3_worker_commit_bytes':12<<30,'device_used_bytes':7<<30},
            'native_attempts_per_stage':1,'automatic_retry':False,
            'pointcloud_scope':'foreground_feature_centres; segmentation precedes SfM; mask consumption must be verified',
            'feature_selection_policy':'selected nonempty object masks only; no whole-image fallback',
            'point_selection':options.point_selection,
            'model_selection':'explicit model_id, otherwise most registered views then points then model_id; never merge components',
            'missing_mask_policy':'exclude unavailable views with an identity-bound subset certificate; never fill missing masks',
            'hole_policy':'preserve gaps; no automatic hole filling; physical scale remains unknown',
            'final_goal':'viewable surface + reopened Blender + verified faceted FCStd/STEP + report' if options.target=='cad'
                         else 'viewable reconstructed surface + independently reopened Blender file + report' if options.target=='blender'
                         else 'originals + real segmentation + available sparse pointcloud + report'}


def _invoke(root: Path, run: Path, stage: str, argv: list[str], *, seconds: int, memory: int,
            additional_roots: list[Path] | None = None) -> dict:
    from integrations.demo_pipeline.supervision import supervise
    logs = run/'logs'/stage; logs.mkdir(parents=True,exist_ok=False)
    temp = logs/'tmp';temp.mkdir()
    env = {k:'2' for k in THREADS}
    env.update(PYTHONDONTWRITEBYTECODE='1',TEMP=str(temp),TMP=str(temp))
    cancelled = False
    def request_cancel(signum, frame):
        # Defer delivery until the supervisor owns the newly created process.
        nonlocal cancelled
        cancelled = True
    previous_handler = signal.signal(signal.SIGINT, request_cancel)
    try:
        entry = supervise(argv,cwd=root,env_overrides=env,log_path=logs/'supervision.jsonl',
            stdout_path=logs/'stdout.txt',stderr_path=logs/'stderr.txt',wall_clock_seconds=seconds,
            job_memory_limit_bytes=memory,run_dir=run,run_dir_threshold_bytes=2<<30,
            total_roots=[run]+list(additional_roots or []),total_threshold_bytes=3<<30,
            poll_seconds=0.5,reclamation_seconds=15,label='photo-demo-'+stage,
            script_sha256=sha(Path(__file__)),cancel_requested=lambda: cancelled)
        entry['cancellation_requested'] = cancelled
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    save(logs/'ENTRY.json',entry)
    return entry


def entry_code(entry: dict) -> int:
    if entry.get('reclaim_confirmed') is not True: return 4
    if entry.get('cancellation_requested') or entry.get('termination_reason') == 'user_cancelled': return 130
    if entry.get('errors') or entry.get('log_errors'): return 4
    if entry.get('status') != 'completed': return 5 if entry.get('termination_reason') in ('wall_clock_exceeded','run_dir_threshold','total_threshold') else 4
    code = entry.get('root_exit_code')
    return code if type(code) is int else 4


def _safe_child(root: Path, relative: str) -> Path:
    target=(root/relative).resolve(strict=True)
    if not target.is_relative_to(root.resolve()): raise ValueError('artifact path escapes run')
    return target


def validate_view_geometry(output: Path, result: dict, source_run: Path, model_id: str) -> dict:
    """Authenticate the complete numerical diagnostic before publishing links."""
    output=Path(output).resolve();source_run=Path(source_run).resolve(strict=True)
    def bound(ref, base):
        if not isinstance(ref,dict):raise ValueError('diagnostic binding must be an object')
        path=Path(ref['path']).resolve(strict=True)
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size!=ref.get('bytes') or sha(path)!=ref.get('sha256'):
            raise ValueError('diagnostic binding path/size/hash differs')
        return path
    result_path=output/'RESULT.json'
    if read(result_path)!=result:raise ValueError('diagnostic returned result differs from saved RESULT')
    if (result.get('schema')!='native-view-geometry-v1' or result.get('status') not in ('DIAGNOSTIC_COMPLETE','PARTIAL_DIAGNOSTIC')
            or result.get('source_model_id')!=model_id or result.get('source_sfm_status')!=read(source_run).get('status')
            or result.get('geometry_acceptance') is not False or result.get('scale_status')!='unknown'
            or not isinstance(result.get('counts'),dict)):
        raise ValueError('diagnostic schema/status/model/source or acceptance differs')
    refs=result.get('source_bindings')
    if not isinstance(refs,list):raise ValueError('diagnostic source bindings missing')
    paths=[bound(ref,source_run.parent) for ref in refs]
    if len(paths)!=len(set(paths)):raise ValueError('duplicate diagnostic source binding')
    if source_run not in paths or not {'cameras.bin','images.bin','points3D.bin'}.issubset({p.name for p in paths}):
        raise ValueError('diagnostic native source bindings incomplete')
    from integrations.demo_pipeline.surface import choose_model
    selected,_=choose_model(read(source_run).get('models',[]),model_id)
    binary=(source_run.parent/selected['binary_path']).resolve(strict=True)
    if not binary.is_relative_to(source_run.parent) or {source_run,binary/'cameras.bin',binary/'images.bin',binary/'points3D.bin'}!=set(paths):
        raise ValueError('diagnostic source does not match selected native model')
    artifacts=result.get('artifacts')
    if not isinstance(artifacts,list):raise ValueError('diagnostic artifacts missing')
    paths=[bound(ref,output) for ref in artifacts]
    if len(paths)!=3 or set(paths)!={output/'PLAN.json',output/'TRACKS.json',output/'report.html'}:
        raise ValueError('diagnostic required artifacts differ')
    return {'path':str(result_path),'bytes':result_path.stat().st_size,'sha256':sha(result_path)}


def make_viewer(sfm_run: Path, run: Path) -> dict:
    """Show all usable components separately. Never merge or invent surfaces."""
    from integrations.viewer.export import export_html
    record = read(sfm_run/'run.json')
    if record.get('status') not in ('SUCCESS','PARTIAL'):
        return {'status':record.get('status','FAILED'),'stats':record.get('counts',{}),'summary':'重建没有产生可用三维分量。'}
    models = [m for m in record.get('models',[]) if m.get('usable') is True and m.get('roundtrip_verified') is True]
    if not models: return {'status':'NO_MODEL','summary':'没有通过原生往返检查的可用分量。'}
    if len(models)>8: return {'status':'VIEWER_LIMIT','summary':'分量超过查看器上限；保留原始成果，未自动删选。'}
    folder=run/'pointcloud';folder.mkdir(exist_ok=False)
    scenes=[]; total=0; components=[]
    for index, model in enumerate(models):
        desc_path=_safe_child(sfm_run,model['descriptions_path'])
        desc_hash=sha(desc_path); desc=read(desc_path)['source']; points=desc['points3D']
        if not isinstance(points,dict) or len(points)!=model['num_points3D'] or not points: raise ValueError('point count differs from accepted native model')
        total+=len(points)
        if total>100000: raise ValueError('viewer point budget exceeded; no silent sampling')
        ply=folder/f'component_{index:02d}.ply'
        header=['ply','format ascii 1.0',f'element vertex {len(points)}','property double x','property double y','property double z',
                'property uchar red','property uchar green','property uchar blue','end_header']
        values=[]
        for key in sorted(points,key=int):
            point=points[key]; xyz=point['xyz']; rgb=point['color']
            if len(xyz)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in xyz): raise ValueError('non-finite point')
            if len(rgb)!=3 or any(type(x) is not int or not 0<=x<=255 for x in rgb):raise ValueError('invalid native RGB')
            values.append(' '.join(format(float(x),'.17g') for x in xyz)+' '+' '.join(map(str,rgb)))
        ply.write_text('\n'.join(header+values)+'\n',encoding='ascii')
        if sha(desc_path)!=desc_hash:raise ValueError('native description changed during export')
        evidence={'model_id':model['model_id'],'description':str(desc_path),'description_sha256':desc_hash,
                  'ply_sha256':sha(ply),'num_points':len(points),'num_registered_images':model['num_registered_images'],
                  'source_geometry':'native SfM source description; unchanged XYZ/RGB, no faces, scale unknown'}
        evidence_path=folder/f'component_{index:02d}.json';save(evidence_path,evidence)
        components.append(evidence)
        scenes.append({'ply':ply.name,'sha256':sha(ply),'title':f'场景分量 {model["model_id"]}',
            'data_kind':'real_photos','declared_unit':None,'note':'真实照片稀疏重建，包含背景；未知真实尺度；分量独立，未融合。',
            'evidence':[evidence_path.name]})
    spec=folder/'scenes.json';save(spec,{'schema':'ply-viewer-scenes-v1','scenes':scenes})
    result=export_html(spec,folder/'viewer.html');save(folder/'VIEWER_RESULT.json',result)
    return {'status':'SUCCESS','viewer_path':'pointcloud/viewer.html','ply_path':'pointcloud/component_00.ply',
        'title':'场景稀疏点云','summary':f'{len(models)} 个独立分量，共 {total} 个点。',
        'stats':{'registered_images':record['counts']['registered_unique_images'],
                 'input_images':record['counts']['unique_images'],'points3d':total,'components':len(models)},
        'components':components,'limitations':['可能含背景；分割未用于过滤点云。','尺度未知，尚无网格或 CAD 实体。'],
        'evidence_paths':['pointcloud/VIEWER_RESULT.json']}


def report_existing(run: Path) -> Path:
    from integrations.demo_pipeline.report import render_report
    return render_report(run.resolve(strict=True))


def verify_geometry_outputs(outcome: dict, geometry_root: Path) -> None:
    """A stage status alone cannot publish missing, stale or unrelated files."""
    geometry_root=geometry_root.resolve()
    def artifact(ref):
        if not isinstance(ref,dict) or not isinstance(ref.get('path'),str):
            raise RuntimeError('geometry artifact has no path binding')
        path=Path(ref['path']).resolve(strict=True)
        if not path.is_relative_to(geometry_root) or not path.is_file() or path.stat().st_size==0:
            raise RuntimeError('geometry artifact is empty or outside this run')
        if sha(path)!=ref.get('sha256'):raise RuntimeError('geometry output hash differs')
    surface=outcome.get('surface',{})
    if surface.get('status')=='SUCCESS':
        refs=surface.get('artifacts')
        if not isinstance(refs,list) or not refs:raise RuntimeError('surface success without artifacts')
        for ref in refs:artifact(ref)
    blender=outcome.get('blender',{})
    if blender.get('status')=='SUCCESS':
        if surface.get('status')!='SUCCESS' or blender.get('reopen_verified') is not True:
            raise RuntimeError('Blender success without surface and independent reopen')
        artifact(blender)



def run_post_review(root, run, python, invoke, builder=None):
    """Independent diagnostic stage; its outcome never upgrades geometry acceptance."""
    output=run/'post_review'
    argv=[str(python),'-B','-X','utf8',str(root/'scripts/review_photo_model.py'),
          '--pipeline',str(run),'--output',str(output)]
    if builder is None:
        entry=invoke(root,run,'post_review',argv,seconds=120,memory=1<<30,additional_roots=[])
    else:
        entry=builder(pipeline=run,output=output)
    code=entry_code(entry)
    if entry.get('reclaim_confirmed') is not True:
        raise ValueError('post-review process reclamation not confirmed')
    if code not in (0,1,2):raise ValueError('post-review unexpected exit code')
    path=output/'RESULT.json';page=output/'index.html'
    if path.is_symlink() or page.is_symlink() or path.resolve().parent!=output.resolve() or page.resolve().parent!=output.resolve():
        raise ValueError('post-review output path escapes diagnostic directory')
    result=read(path)
    expected={0:'DIAGNOSTIC_COMPLETE',1:'INSUFFICIENT_DATA',2:'FAILED'}
    if not isinstance(result,dict) or result.get('status')!=expected[code]:
        raise ValueError('post-review result disagrees with process exit')
    bind=lambda p: dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
    return dict(status=result['status'],exit_code=code,reclaim_confirmed=True,
                record=bind(path),report=bind(page),geometry_acceptance=False,
                started_by='automatic_offline_post_review')

def run_point_review(root, run, python, invoke, builder=None):
    """Single bounded diagnostic; never grants geometry acceptance."""
    output=run/'point_review'
    argv=[str(python),'-B','-X','utf8',str(root/'scripts/review_point_sequence.py'),
          '--pipeline',str(run),'--output',str(output)]
    entry=(invoke(root,run,'point_review',argv,seconds=180,memory=2<<30,additional_roots=[])
           if builder is None else builder(pipeline=run,output=output))
    code=entry_code(entry)
    detail=dict(status='FAILED',exit_code=code,reclaim_confirmed=entry.get('reclaim_confirmed'),
                geometry_accepted=False,started_by='automatic_point_review')
    resource=(code==5 or entry.get('termination_reason') in
              ('job_memory_limit','memory_limit_exceeded','memory_limit','resource_limit')
              or entry.get('root_exit_code') in (0xC0000017,0xC000009A))
    detail['blocks_post_review']=resource or code==130 or entry.get('reclaim_confirmed') is not True
    if detail['blocks_post_review']:
        detail.update(status='CANCELLED' if code==130 else ('RESOURCE_STOPPED' if resource else 'UNRECLAIMED'),
                      reason=entry.get('termination_reason') or 'diagnostic process not reclaimed')
        return detail
    if code not in (0,1,2):raise ValueError('point-review unexpected exit code')
    path=output/'RESULT.json';page=output/'index.html'
    for item in (path,page):
        if item.is_symlink() or item.resolve().parent!=output.resolve():
            raise ValueError('point-review output path escapes diagnostic directory')
    result=read(path)
    if result.get('status')!={0:'AUTO_POINT_REVIEW_COMPLETE',1:'INSUFFICIENT_DATA',2:'FAILED'}[code] or result.get('geometry_accepted') is not False:
        raise ValueError('point-review result disagrees with process exit or acceptance contract')
    bind=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
    detail.update(status=result['status'],record=bind(path),report=bind(page))
    return detail

def run_pipeline(options: Options, *, root: Path = ROOT, executor=None, renderer=None, surface_builder=None, research_builder=None, cad_builder=None, view_geometry_builder=None, post_review_builder=None, point_review_builder=None) -> tuple[Path,int]:
    """The injection seams are only for offline orchestration tests."""
    frozen=plan(options,root)
    source=Path(frozen['source']['path'])
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    run=(options.output or root/'outputs/photo_demo'/run_id).expanduser().resolve()
    if not run.is_relative_to((root/'outputs').resolve()) or run==(root/'outputs').resolve():raise ValueError('输出必须是项目 outputs 内的新目录。')
    if run==source or run.is_relative_to(source) or source.is_relative_to(run):raise ValueError('输入与输出不能重叠。')
    run.mkdir(parents=True,exist_ok=False)
    save(run/'PLAN.json',frozen)
    save(run/'INPUT_INVENTORY.json',{'source':str(source),'object':options.prompt,'data_kind':'real_photos','files':frozen['source']['files']})
    state={'schema':'photo-demo-pipeline-v1','run_id':run_id,'started_utc':utc(),'status':'RUNNING','source':str(source),
           'source_kind':frozen['source']['kind'],'prompt':options.prompt,'object_id':options.object_id,'target':frozen['target'],
           'stages':{s:{'status':'NOT_RUN'} for s in frozen['stages']},'errors':[],'exit_code':None}
    report_data={'title':'照片与视频重建演示','object_label':options.prompt,'source':str(source),'source_kind':frozen['source']['kind'],
        'pointcloud':{'status':'NOT_RUN'},'sfm_masks_applied':False,'sfm_masks_requested':True,
        'surface':{'status':'NOT_RUN'},'blender':{'status':'NOT_RUN'},
        'research':{'status':'NOT_RUN'},'view_geometry':{'status':'NOT_RUN'},'post_review':{'status':'NOT_RUN'},'point_review':{'status':'NOT_RUN'},
        'limitations':['先分割目标，再在掩膜内选点；分割误差与错误对应仍需检查，掩膜不是几何真值。','表面允许孔洞；未标定物理尺度，不代表封闭或参数化 CAD 实体。']}
    invoke=executor or _invoke
    render=renderer or report_existing
    failure_code=None; abort_heavy=False; user_cancelled=False; sfm_path=None; sfm={}; product={}
    def persist():
        state['updated_utc']=utc();save(run/'pipeline.json',state);save(run/'REPORT_DATA.json',report_data)
    def stage(name, argv, seconds, memory, extra=None):
        nonlocal abort_heavy, failure_code, user_cancelled
        state['stages'][name]={'status':'RUNNING','started_utc':utc(),'argv':argv};persist()
        print(f'[{name}] started; logs: {run / "logs" / name}',flush=True)
        entry=invoke(root,run,name,argv,seconds=seconds,memory=memory,additional_roots=extra or [])
        code=entry_code(entry)
        if (code!=130 and (entry.get('termination_reason') in
                ('job_memory_limit','memory_limit_exceeded','memory_limit','resource_limit')
                or entry.get('root_exit_code') in (0xC0000017,0xC000009A))):
            code=5
        if code >= 2 and failure_code is None:
            failure_code=code
        if code==130:user_cancelled=True
        if code in (5,130):
            abort_heavy=True
            state['errors'].append('任务已取消，停止后续阶段。' if code==130 else '资源或时间预算触发，停止后续原生阶段。')
        state['stages'][name].update(status='EXECUTED',exit_code=code,finished_utc=utc(),
                                    entry_path=str(run/'logs'/name/'ENTRY.json'),reclaim_confirmed=entry.get('reclaim_confirmed'))
        if entry.get('reclaim_confirmed') is not True:
            abort_heavy=True;failure_code=4
        print(f'[{name}] exit={code}',flush=True)
        persist();return code
    def check_sources():
        if any(sha(Path(row['path']))!=row['sha256'] for row in frozen['source']['files']):
            raise ValueError('输入文件在运行中发生变化。')
    try:
        if not frozen['dependencies']['project_python']:raise ValueError('项目 Python 环境不存在。')
        if not frozen['dependencies']['git_exe']:
            failure_code=3
            raise ValueError('SAM3源码校验需要已有Git；请用 --git-exe 指定绝对路径。输入与SfM未启动。')
        if frozen['target'] in ('blender','cad') and not all(frozen['dependencies'].get(key) for key in ('openmvs','blender')):
            failure_code=3
            raise ValueError('完整三维输出需要现有 OpenMVS 与 Blender；用参数指定本地工具，或显式选择 --target report。')
        if frozen['target']=='cad' and not frozen['dependencies'].get('freecad_python'):
            failure_code=3
            raise ValueError('CAD 输出需要现有 FreeCAD Python；用 --freecad-python 指定本地工具。')
        if frozen['source']['kind']=='video' and not all(frozen['dependencies'][k] for k in ('ffmpeg','ffprobe')):
            raise ValueError('视频输入需要本地 ffmpeg/ffprobe；可用参数指定现有可执行文件。')
        dataset_id='demo_'+run_id.lower()
        argv=[frozen['python'],'-B',str(root/'scripts/prepare_object.py'),frozen['source']['kind'],
              '--source',str(source),'--dataset-id',dataset_id,'--data-kind','real','--output',str(run/'input')]
        if frozen['source']['kind']=='video':
            argv += ['--count',str(options.frames),'--ffmpeg',frozen['video_tools']['ffmpeg'],'--ffprobe',frozen['video_tools']['ffprobe']]
            if frozen['all_frames']:argv+=['--all-frames']
        code=stage('input',argv,180,2<<30,[root/'data/real'/dataset_id])
        prepared=read(run/'input/RESULT.json') if (run/'input/RESULT.json').is_file() else {}
        if code!=0 or prepared.get('status')!='INPUTS_PREPARED':
            state['stages']['input']['status']='FAILED';failure_code=code or 2
            raise ValueError(prepared.get('error','input preparation failed; see logs/input'))
        state['stages']['input']['status']='SUCCESS'
        io_file=Path(prepared['photo_io']['run']['path']).resolve(strict=True)
        if not io_file.is_relative_to(run/'input') or sha(io_file)!=prepared['photo_io']['run']['sha256']:
            raise ValueError('prepared photo-io identity mismatch')
        io_run=io_file.parent
        indices=[r['input_index'] for r in prepared['photo_io']['identities']]
        if len(indices)!=len(set(indices)) or any(type(i) is not int for i in indices):raise ValueError('invalid prepared indices')
        if not 2<=len(indices)<=frozen['video_frame_limit']:raise ValueError('prepared count exceeds selected profile')
        persist()
        if not abort_heavy:
            output=run/'segmentation';output.mkdir()
            config={k:frozen[k] for k in ('sam3_python','checkpoint','checkpoint_sha256','object_id','git_exe')}
            config.update(root=str(root),io_run=str(io_run),output_dir=str(output),indices=indices,
                          worker_package=frozen['sam3_package'],text_prompt=frozen['prompt'])
            if all(frozen['dependencies'].get(k) for k in ('sam3_python','sam3_package','checkpoint')):
                if len(indices)<=96:
                    save(output/'CONFIG.json',config)
                    code=stage('segmentation',[frozen['sam3_python'],'-B',str(root/'scripts/demo_segment_worker.py'),
                                              '--config',str(output/'CONFIG.json')],950,14<<30)
                    product=read(output/'PRODUCT_OUTCOME.json') if (output/'PRODUCT_OUTCOME.json').is_file() else {}
                    state['stages']['segmentation'].update(status=product.get('status','FAILED'),counts=product.get('counts'),run_dir=product.get('run_dir'))
                else:
                    from .segmentation_batches import merge_manifests
                    batch_manifests=[];batch_records=[]
                    for offset in range(0,len(indices),96):
                        number=offset//96;batch=output/('batch_%02d'%number);batch.mkdir()
                        batch_config=dict(config,output_dir=str(batch),indices=indices[offset:offset+96])
                        save(batch/'CONFIG.json',batch_config)
                        name='segmentation_batch_%02d'%number
                        code=stage(name,[frozen['sam3_python'],'-B',str(root/'scripts/demo_segment_worker.py'),
                                        '--config',str(batch/'CONFIG.json')],950,14<<30)
                        current=read(batch/'PRODUCT_OUTCOME.json') if (batch/'PRODUCT_OUTCOME.json').is_file() else {}
                        batch_records.append(dict(stage=name,indices=batch_config['indices'],outcome=current))
                        save(output/'BATCHES.json',dict(profile='whole-video-192-v1',batches=batch_records,
                            requested_indices=indices,automatic_retry=False))
                        if code not in (0,1) or current.get('status') not in ('SUCCESS','PARTIAL') or abort_heavy:
                            product=dict(status=current.get('status','FAILED'),counts=current.get('counts'),run_dir=None)
                            break
                        product_root=Path(current['run_dir']).resolve(strict=True)
                        if not product_root.is_relative_to(batch):raise ValueError('batch product escapes its output')
                        batch_manifests.append(product_root/'manifest.json')
                    if len(batch_manifests)==(len(indices)+95)//96:
                        product=merge_manifests(batch_manifests,output/'merged',indices)
                    save(output/'PRODUCT_OUTCOME.json',product)
                    state['stages']['segmentation'].update(status=product.get('status','FAILED'),counts=product.get('counts'),
                        run_dir=product.get('run_dir'),batch_count=len(batch_records),batch_records=batch_records,
                        composition='identity-bound manifest aggregation; separate real worker per batch')
            else:
                state['stages']['segmentation']={'status':'BLOCKED','error':'SAM3 环境、封包或权重不存在；未安装或下载。'}
                failure_code=failure_code or 3
            report_data['segmentation']=state['stages']['segmentation'];persist()
        else:state['errors'].append('停止条件已触发，后续原生阶段没有启动；详见阶段执行记录。')
        if frozen['automatic_recovery'] and not abort_heavy:
            from .automatic_recovery import recover_segmentation
            original_product=product
            product,recovery=recover_segmentation(run=run,root=root,config=config,product=product,
                indices=indices,prompt=frozen['recovery_prompt'],stage=stage,stopped=lambda:abort_heavy)
            state['automatic_segmentation_recovery']=recovery
            report_data['automatic_segmentation_recovery']=recovery
            report_data.setdefault('processing_notes',[]).append(
                '自动分割恢复：'+recovery['status']+'；新增唯一掩膜 '+str(recovery.get('recovered',0))+' 张。原始成功掩膜保留，质量尚未验收。')
            if product is not original_product:
                state['stages']['segmentation_primary']=dict(state['stages']['segmentation'])
                original_record=run/'segmentation/PRODUCT_OUTCOME.json'
                shutil.copyfile(original_record,run/'segmentation/PRIMARY_PRODUCT_OUTCOME.json')
                product.update(stop_state={'latched':False},execution_mode='recovery_composition_no_model_execution',
                    execution_stages=['segmentation_primary','segmentation_recovery'])
                save(original_record,product)
                state['stages']['segmentation']=dict(status=product['status'],exit_code=product['exit_code'],
                    counts=product['counts'],run_dir=product['run_dir'],
                    composition='original OK + uniquely recovered unresolved frames',
                    execution_stages=['segmentation_primary','segmentation_recovery'])
                report_data['segmentation']=state['stages']['segmentation']
            persist()
        foreground_manifest=None
        if not abort_heavy and product.get('status') in ('SUCCESS','PARTIAL'):
            if not product.get('run_dir'):
                raise ValueError('object-first segmentation has no run_dir; whole-image fallback forbidden')
            segment_run=Path(product['run_dir']).resolve(strict=True)
            if not segment_run.is_relative_to((run/'segmentation').resolve()):
                raise ValueError('segmentation run escapes this pipeline output')
            candidate_manifest=segment_run/'manifest.json'
            mask_record=read(candidate_manifest)
            if mask_record.get('status')!=product['status']:
                raise ValueError('segmentation outcome/manifest status mismatch')
            frames=mask_record.get('frames')
            if not isinstance(frames,list):raise ValueError('segmentation manifest has no frames')
            seen=set();selected=[]
            for frame in frames:
                if not isinstance(frame,dict):raise ValueError('segmentation frame must be an object')
                index=frame.get('input_index')
                if type(index) is not int or index not in indices or index in seen:
                    raise ValueError('segmentation frame identity is invalid or duplicated')
                seen.add(index)
                if frame.get('status')=='OK' and isinstance(frame.get('selected_mask'),dict):
                    selected.append(index)
            state['foreground_selection']={'policy':'object-first-v1','candidate_indices':selected,
                'input_count':len(indices),'manifest':str(candidate_manifest),'manifest_sha256':sha(candidate_manifest),
                'validation':'availability only; full identity and mask validation occurs in SfM CLI'}
            if len(selected)>=2:
                foreground_manifest=candidate_manifest
            else:
                state['stages']['sfm']={'status':'NOT_RUN','reason':'INSUFFICIENT_FOREGROUND_VIEWS',
                    'usable_mask_count':len(selected),'whole_image_fallback':False}
        elif not abort_heavy:
            state['stages']['sfm']={'status':'NOT_RUN','reason':'NO_USABLE_SEGMENTATION','whole_image_fallback':False}
        persist()
        if not abort_heavy and foreground_manifest is not None:
            check_sources()
            sfm_output=root/'outputs/delegation/T02A'/('PIPELINE_'+run_id)
            if sfm_output.exists():raise ValueError('SfM output exists')
            code=stage('sfm',[frozen['python'],'-B',str(root/'scripts/reconstruct_object.py'),
                       '--io-run',str(io_run),'--output',str(sfm_output),
                       '--profile',frozen['sfm_profile'],
                       '--object-mask-manifest',str(foreground_manifest),
                       '--point-selection',frozen['point_selection']],720,2<<30,[sfm_output])
            candidates=list(sfm_output.glob('*/run.json'))
            if len(candidates)==1:
                sfm_path=candidates[0];sfm=read(sfm_path)
                foreground=sfm.get('foreground',{})
                report_data['sfm_masks_applied']=foreground.get('mask_consumption_verified') is True
                report_data['foreground']=foreground
                state['stages']['sfm'].update(status=sfm.get('status','FAILED'),run_json=str(sfm_path),counts=sfm.get('counts'))
                if frozen['automatic_recovery'] and code in (0,1) and sfm.get('status') in ('SUCCESS','PARTIAL') and not abort_heavy:
                    from .automatic_recovery import select_camera_solution
                    primary_path=sfm_path
                    sfm_path,recovery=select_camera_solution(root=root,run=run,frozen=frozen,io_run=io_run,
                        manifest=foreground_manifest,sfm_path=sfm_path,stage=stage,stopped=lambda:abort_heavy)
                    state['automatic_camera_recovery']=recovery
                    report_data['automatic_camera_recovery']=recovery
                    report_data.setdefault('processing_notes',[]).append(
                        '自动相机域检查：'+recovery['status']+'；独立对照次数 '+str(recovery['attempts'])+'。数值域有效不代表形状或相机标定正确。')
                    if sfm_path!=primary_path:
                        state['stages']['sfm_primary']=dict(state['stages']['sfm'])
                        sfm=read(sfm_path)
                        state['stages']['sfm']=dict(state['stages']['sfm_constrained'])
                        state['stages']['sfm'].update(status=sfm['status'],run_json=str(sfm_path),counts=sfm.get('counts'),
                            selected_from='sfm_constrained',original_run_json=str(primary_path))
                        foreground=sfm.get('foreground',{})
                        report_data['foreground']=foreground
                        report_data['sfm_masks_applied']=foreground.get('mask_consumption_verified') is True
                report_data['pointcloud']={'status':sfm.get('status','FAILED'),
                    'stats':{'registered_images':sfm.get('counts',{}).get('registered_unique_images',0),'input_images':len(indices),'points3d':0},
                    'summary':'当前配置未产生可用点云。','evidence_paths':[str(sfm_path)]}
                if code in (0,1) and sfm.get('status') in ('SUCCESS','PARTIAL') and not abort_heavy:
                    try:
                        report_data['pointcloud']=make_viewer(sfm_path.parent,run)
                        state['stages']['viewer']={'status':report_data['pointcloud']['status']}
                    except Exception as exc:
                        state['stages']['viewer']={'status':'FAILED','error':str(exc)}
                        report_data['pointcloud']['summary']='原生成果已保留，查看器导出失败：'+str(exc)
                if sfm.get('point_selection',{}).get('status') == 'VERIFIED':
                    try:
                        from integrations.demo_pipeline.selected_point_report import build_report
                        selection_report=build_report(sfm_path.parent,run/'selected_points')
                        state['stages']['point_selection_report']={
                            'status':'DIAGNOSTIC_READY','result':selection_report,
                            'index':'selected_points/index.html','geometry_acceptance':False}
                        report_data['pointcloud']['evidence_paths'].append(str(run/'selected_points/index.html'))
                    except Exception as exc:
                        state['stages']['point_selection_report']={'status':'FAILED','error':str(exc)}
            else:state['stages']['sfm']['status']='FAILED'
            persist()
        if frozen['view_geometry_audit']:
            detail={'status':'NOT_RUN','reason':'需要可用原生 SfM 模型。','geometry_acceptance':False,'scale_status':'unknown'}
            state['stages']['view_geometry']=detail
            if not abort_heavy and sfm_path is not None and sfm.get('status') in ('SUCCESS','PARTIAL'):
                try:
                    from integrations.demo_pipeline.surface import choose_model
                    selected, selection=choose_model(sfm.get('models',[]),options.model_id)
                    detail.update(status='RUNNING',selected_model_id=str(selected['model_id']),selection=selection,
                                  source_run=str(sfm_path),source_sfm_status=sfm['status'])
                    persist()
                    binary=(sfm_path.parent/selected['binary_path']).resolve(strict=True)
                    if not binary.is_relative_to(sfm_path.parent.resolve()):raise ValueError('selected binary source escapes SfM run')
                    expected_sources={str(path.resolve()):(path.stat().st_size,sha(path)) for path in [sfm_path]+[binary/name for name in ('cameras.bin','images.bin','points3D.bin')]}
                    diagnostic_output=run/'view_geometry'
                    if view_geometry_builder is not None:
                        result=view_geometry_builder(sfm_run=sfm_path.parent,model_id=str(selected['model_id']),output=diagnostic_output)
                    else:
                        argv=[frozen['python'],'-B','-X','utf8',str(root/'scripts/audit_view_geometry.py'),
                              '--sfm-run',str(sfm_path.parent),'--model-id',str(selected['model_id']),'--output',str(diagnostic_output)]
                        detail['argv']=argv
                        entry=invoke(root,run,'view_geometry',argv,seconds=90,memory=1<<30,additional_roots=[])
                        diagnostic_code=entry_code(entry)
                        detail.update(exit_code=diagnostic_code,reclaim_confirmed=entry.get('reclaim_confirmed'))
                        if (diagnostic_code in (5,130) or entry.get('reclaim_confirmed') is not True
                                or entry.get('termination_reason') in ('job_memory_limit','memory_limit_exceeded','memory_limit','resource_limit')
                                or entry.get('root_exit_code') in (0xC0000017,0xC000009A)):
                            abort_heavy=True;failure_code=diagnostic_code or 4
                            state['errors'].append('几何诊断取消、资源预算或回收失败；停止后续原生阶段。')
                        if diagnostic_code!=0:raise ValueError('diagnostic process failed: '+str(diagnostic_code))
                        result=read(diagnostic_output/'RESULT.json')
                    actual_sources={str(Path(ref['path']).resolve()):(ref.get('bytes'),ref.get('sha256')) for ref in result.get('source_bindings',[])}
                    if actual_sources!=expected_sources:raise ValueError('diagnostic source changed after stage started')
                    result_ref=validate_view_geometry(diagnostic_output,result,sfm_path,str(selected['model_id']))
                    detail.update(status=result['status'],counts=result['counts'],result=result_ref,
                                  artifacts=result['artifacts'],source_bindings=result['source_bindings'])
                except Exception as exc:
                    detail.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                detail['finished_utc']=utc()
            report_data['view_geometry']=detail;persist()
        if (not abort_heavy and sfm_path is not None and sfm.get('status') in ('SUCCESS','PARTIAL')
                and (not frozen['automatic_recovery'] or state.get('automatic_camera_recovery',{}).get('surface_ready') is True)
                and product.get('status') in ('SUCCESS','PARTIAL') and product.get('run_dir')):
            # Commit upstream identity while this producer is still running.
            # A missing checkpoint disables reuse, not the existing one-shot path.
            try:
                check_sources()
                from integrations.demo_pipeline.resume import write_resume_checkpoint
                checkpoint = write_resume_checkpoint(run, root=root)
                state['resume_checkpoint'] = {'status':'COMMITTED', **checkpoint}
            except Exception as exc:
                state['resume_checkpoint'] = {'status':'UNAVAILABLE','reason':f'{type(exc).__name__}: {exc}'}
            persist()
        if frozen['target'] in ('blender','cad'):
            eligible=(not abort_heavy and sfm_path is not None and sfm.get('status') in ('SUCCESS','PARTIAL')
                      and (not frozen['automatic_recovery'] or state.get('automatic_camera_recovery',{}).get('surface_ready') is True)
                      and product.get('status') in ('SUCCESS','PARTIAL') and product.get('run_dir'))
            if eligible:
                check_sources()
                segment_run=Path(product['run_dir']).resolve(strict=True)
                if not segment_run.is_relative_to((run/'segmentation').resolve()):
                    raise ValueError('segmentation run escapes this pipeline output')
                manifest=segment_run/'manifest.json'
                dataset_record=prepared['photo_io']['manifest']
                dataset_manifest=Path(dataset_record['path']).resolve(strict=True)
                if not dataset_manifest.is_relative_to((run/'input').resolve()) or sha(dataset_manifest)!=dataset_record['sha256']:
                    raise ValueError('prepared dataset manifest identity mismatch')
                if surface_builder is None:
                    from integrations.demo_pipeline.surface import complete_surface
                    builder=complete_surface
                else: builder=surface_builder
                state['stages']['surface']={'status':'RUNNING','started_utc':utc()};persist()
                outcome=builder(sfm_run=sfm_path.parent,io_run=io_run,dataset_manifest=dataset_manifest,
                    mask_manifests=[manifest],run=run/'geometry',root=root,python=Path(frozen['python']),
                    tool_root=Path(frozen['mvs_tool_root']),blender_exe=Path(frozen['blender_exe']),
                    model_id=frozen['model_id'],stage=stage)
                if not isinstance(outcome,dict):raise ValueError('surface builder returned no result')
                verify_geometry_outputs(outcome,run/'geometry')
                for name in ('surface','blender'):
                    detail=outcome.get(name,{'status':'NOT_RUN'})
                    state['stages'][name]=detail;report_data[name]=detail
                state['geometry_steps']=outcome.get('stages',{})
                code=outcome.get('exit_code')
                if type(code) is not int:raise ValueError('surface builder omitted exit code')
                if code>=2:failure_code=failure_code or code
                if outcome.get('error'):state['errors'].append(outcome['error'])
                persist()
            else:
                reason='前置阶段未产生可用模型与正式分割结果、相机域检查未通过，或停止条件已触发。'
                for name in ('surface','blender'):
                    state['stages'][name]={'status':'NOT_RUN','reason':reason};report_data[name]=state['stages'][name]
        if frozen['research_code6']:
            selected=state['stages'].get('surface',{}).get('selected_model_id')
            if (not abort_heavy and not failure_code and sfm_path is not None and selected is not None
                    and state['stages'].get('surface',{}).get('status')=='SUCCESS'
                    and state['stages'].get('blender',{}).get('status')=='SUCCESS'):
                detail={'status':'RUNNING','selected_model_id':str(selected),'started_utc':utc()}
                state['stages']['research']=detail;persist()
                try:
                    research_output=run/'research'
                    if research_builder is not None:
                        result=research_builder(sfm_run=sfm_path.parent,model_id=str(selected),output=research_output)
                    else:
                        argv=[frozen['python'],'-B','-X','utf8',str(root/'scripts/export_research_code6.py'),
                              '--sfm-run',str(sfm_path.parent),'--model-id',str(selected),'--output',str(research_output)]
                        detail['argv']=argv
                        entry=invoke(root,run,'research',argv,seconds=30,memory=1<<30,additional_roots=[])
                        detail.update(exit_code=entry_code(entry),entry_path=str(run/'logs/research/ENTRY.json'),
                                      reclaim_confirmed=entry.get('reclaim_confirmed'))
                        result=read(research_output/'RESULT.json') if (research_output/'RESULT.json').is_file() else {}
                        if entry_code(entry)!=0:
                            detail.update(status=result.get('status','FAILED'),error=result.get('error','research stage failed; geometry retained'))
                            if detail['status']=='SUCCESS':detail['status']='FAILED'
                    if not isinstance(result,dict):raise ValueError('research result missing')
                    if detail.get('exit_code',0)==0:
                        detail.update(status=result.get('status','FAILED'))
                    detail.update(source_sfm_status=result.get('source_sfm_status'),counts=result.get('counts'),limitations=result.get('limitations',[]))
                    if detail['status']=='SUCCESS':
                        from integrations.demo_pipeline.research_code6 import binding
                        if (result.get('complete') is not True or str(result.get('source_model_id'))!=str(selected)
                                or result.get('algorithm_called')!='kozlov_lab.stereo.official.indexed_code3d.indexed_body_code'):
                            raise ValueError('research success has no matching complete algorithm call')
                        refs=result.get('artifacts')
                        if not isinstance(refs,list) or not refs:raise ValueError('research artifacts missing')
                        for ref in refs:
                            artifact_path=Path(ref['path']).resolve(strict=True)
                            if not artifact_path.is_relative_to(research_output.resolve()) or sha(artifact_path)!=ref.get('sha256'):
                                raise ValueError('research artifact identity differs')
                        if not any(Path(ref['path']).resolve()==sfm_path.resolve() and sha(sfm_path)==ref.get('sha256')
                                   for ref in result.get('source_bindings',[])):
                            raise ValueError('research source SfM binding differs')
                        detail['result']=binding(research_output/'RESULT.json')
                        detail['artifacts']=result.get('artifacts',[])
                    if result.get('error'):detail['error']=result['error']
                except Exception as exc:
                    detail.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                detail['finished_utc']=utc();report_data['research']=detail;persist()
            else:
                detail={'status':'NOT_RUN','reason':'研究侧支需要成功表面/Blender以及已选原生模型；不替代几何交付。'}
                state['stages']['research']=detail;report_data['research']=detail;persist()
        if frozen['target']=='cad':
            eligible=(not abort_heavy and not failure_code
                      and state['stages'].get('surface',{}).get('status')=='SUCCESS'
                      and state['stages'].get('blender',{}).get('status')=='SUCCESS'
                      and state['stages']['blender'].get('reopen_verified') is True)
            if eligible:
                check_sources()
                verify_geometry_outputs(state['stages'],run/'geometry')
                state.update(status='GEOMETRY_READY_FOR_CAD',geometry_deliverable_ready=True,deliverable_ready=False)
                persist()
                checkpoint=run/'CAD_SOURCE.json'
                geometry_record={'schema':'photo-cad-source-v1','source_run':str(run),
                    'status':'GEOMETRY_READY_FOR_CAD','geometry_deliverable_ready':True,
                    'target':'cad','deliverable_ready':False,
                    'stages':{name:state['stages'][name] for name in ('surface','blender')}}
                with checkpoint.open('x',encoding='utf-8') as stream:
                    json.dump(geometry_record,stream,ensure_ascii=False,allow_nan=False,indent=2)
                output=run/'cad'
                state['stages']['cad']={'status':'RUNNING','started_utc':utc()};persist()
                if cad_builder is not None:
                    detail=cad_builder(run=run,output=output,geometry_record=checkpoint,
                        python=Path(frozen['python']),freecad_python=Path(frozen['freecad_python']),stage=stage)
                else:
                    argv=[frozen['python'],'-B','-X','utf8',str(root/'scripts/continue_photo_cad.py'),
                          '--run',str(run),'--output',str(output),'--geometry-record',str(checkpoint),
                          '--python',frozen['python'],'--freecad-python',frozen['freecad_python']]
                    code=stage('cad_continuation',argv,240,3<<30)
                    if code==0:
                        from integrations.demo_pipeline.cad import verify_cad_output
                        detail={'status':'SUCCESS','exit_code':0,**verify_cad_output(output,run)}
                        detail['status']='SUCCESS'
                    else:
                        reason=read(output/'RESULT.json').get('error') if (output/'RESULT.json').is_file() else None
                        detail={'status':'FAILED','exit_code':code,'output':str(output),
                                'error':reason or 'CAD continuation failed; Blender retained.'}
                if not isinstance(detail,dict):raise ValueError('CAD continuation returned no result')
                if detail.get('output',str(output))!=str(output):raise ValueError('CAD result belongs to another output')
                detail['output']=str(output)
                code=detail.get('exit_code')
                if type(code) is not int:raise ValueError('CAD continuation omitted exit code')
                if detail.get('status')=='SUCCESS' and code==0:
                    refs=detail.get('artifacts',[])
                    if not isinstance(refs,list) or len(refs)!=2:raise ValueError('CAD success requires FCStd and STEP bindings')
                    expected={str((output/'cad/cad'/name).resolve()) for name in ('model.FCStd','model.step')}
                    if {str(Path(row['path']).resolve()) for row in refs}!=expected:raise ValueError('CAD output roles or paths differ')
                    for row in refs:
                        path=Path(row['path']).resolve(strict=True)
                        if path.stat().st_size!=row.get('bytes') or sha(path)!=row.get('sha256'):
                            raise ValueError('CAD artifact bytes or hash differ')
                else:
                    detail['status']='FAILED';failure_code=failure_code or (code if code>=2 else 4)
                    state['errors'].append(detail.get('error','CAD目标未完成；已生成的Blender文件保留。'))
                state['stages']['cad']=detail;report_data['cad']=detail;persist()
            else:
                state['stages']['cad']={'status':'NOT_RUN','reason':'CAD需要已验证表面与Blender；未补洞或重跑。'}
                failure_code=failure_code or 4
    except KeyboardInterrupt:
        user_cancelled=True
        state['errors'].append('用户取消运行；已存在的成果保留。');failure_code=130
    except Exception as exc:
        state['errors'].append(f'{type(exc).__name__}: {exc}')
        failure_code=failure_code or (2 if isinstance(exc,(ValueError,FileNotFoundError)) else 4)
        if frozen['target']=='cad' and state['stages'].get('cad',{}).get('status')=='RUNNING':
            state['stages']['cad'].update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            if any(sha(Path(row['path']))!=row['sha256'] for row in frozen['source']['files']):
                state['errors'].append('输入文件在运行中发生变化。');failure_code=2
        except OSError as exc:
            state['errors'].append('无法复核原素材：'+str(exc));failure_code=2
        for detail in state['stages'].values():
            if detail.get('status')=='RUNNING':detail['status']='INTERRUPTED'
        report_data['pipeline_errors']=state['errors'];report_data['stage_statuses']=state['stages'];persist()
        try:
            from integrations.demo_pipeline.quality import build_quality
            quality = build_quality(run, state)
            quality_path = run/'QUALITY.json'
            save(quality_path, quality)
            state['quality_reference'] = {'path': str(quality_path), 'sha256': sha(quality_path)}
            state['object_goal_achieved'] = quality['object_goal_achieved']
            persist()
        except Exception as exc:
            state['object_goal_achieved'] = False
            state['errors'].append('无法保存独立外形状态：'+str(exc));failure_code=4
            persist()
        if frozen['post_review']:
            if user_cancelled or failure_code==130:point_detail={'status':'NOT_RUN_CANCELLED'}
            elif failure_code==5:point_detail={'status':'NOT_RUN_RESOURCE_LIMIT'}
            elif any(d.get('reclaim_confirmed') is False for d in state['stages'].values()):point_detail={'status':'NOT_RUN_UNRECLAIMED'}
            elif frozen['source']['kind']!='video':point_detail={'status':'NOT_APPLICABLE','reason':'Video temporal identity required'}
            elif not state['stages'].get('segmentation',{}).get('run_dir'):point_detail={'status':'NOT_AVAILABLE','reason':'No bound segmentation run'}
            else:
                try:point_detail=run_point_review(root,run,frozen['python'],invoke,point_review_builder)
                except Exception as exc:point_detail={'status':'FAILED','error':f'{type(exc).__name__}: {exc}'}
            state['stages']['point_review']=point_detail;report_data['point_review']=point_detail;persist()
            if point_detail.get('blocks_post_review'):
                failure_code={'RESOURCE_STOPPED':5,'CANCELLED':130,'UNRECLAIMED':4}[point_detail['status']]
                if failure_code==130:user_cancelled=True
            detail = {'status':'NOT_RUN_CANCELLED'} if user_cancelled or failure_code==130 else (
                {'status':'NOT_RUN_RESOURCE_LIMIT','reason':'Existing stage exhausted its resource budget; no diagnostic process started.'}
                if failure_code==5 else (
                {'status':'NOT_RUN_UNRECLAIMED'} if any(d.get('reclaim_confirmed') is False for d in state['stages'].values()) else None))
            if detail is None and point_detail.get('blocks_post_review'):
                blocked_status={'RESOURCE_STOPPED':'NOT_RUN_RESOURCE_LIMIT','CANCELLED':'NOT_RUN_CANCELLED',
                                'UNRECLAIMED':'NOT_RUN_UNRECLAIMED'}[point_detail['status']]
                detail={'status':blocked_status,'reason':'Point review stopped; no further diagnostic process started.'}
                if point_detail['status']=='RESOURCE_STOPPED':failure_code=5
                elif point_detail['status']=='CANCELLED':failure_code=130;user_cancelled=True
                else:failure_code=4
            if detail is None:
                try:
                    detail=run_post_review(root,run,frozen['python'],invoke,post_review_builder)
                except Exception as exc:
                    detail={'status':'FAILED','error':f'{type(exc).__name__}: {exc}'}
            state['stages']['post_review']=detail;report_data['post_review']=detail;persist()
        try:
            report=render(run);state['stages']['report']={'status':'SUCCESS','path':str(report)}
        except Exception as exc:
            state['stages']['report']={'status':'FAILED','error':str(exc)};failure_code=4
        state['geometry_deliverable_ready']=(frozen['target'] in ('blender','cad')
            and state['stages']['surface'].get('status')=='SUCCESS'
            and state['stages']['blender'].get('status')=='SUCCESS'
            and state['stages']['blender'].get('reopen_verified') is True)
        state['deliverable_ready']=(state['geometry_deliverable_ready'] and not failure_code
            and (frozen['target']=='blender' or (frozen['target']=='cad' and state['stages']['cad'].get('status')=='SUCCESS')))
        if frozen['target']=='cad':state['target_achieved']=state['deliverable_ready']
        success=(all(state['stages'][s].get('status')=='SUCCESS' for s in frozen['stages'] if s not in ('research','view_geometry','point_review','post_review'))
                 and (frozen['target']=='report' or state['deliverable_ready']))
        state['exit_code']=failure_code or (0 if success else 1)
        state['status']='SUCCESS' if success and not failure_code else 'PARTIAL' if state['stages']['input']['status']=='SUCCESS' else 'FAILED'
        state['finished_utc']=utc();persist()
    return run,state['exit_code']
