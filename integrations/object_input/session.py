"""Common input receipt; successful preparation is not reconstruction success."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import unicodedata
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import PIL

from kozlov_lab.io.errors import FrameIssue
from kozlov_lab.io.hashing import compute_frame_id, source_tree_sha256
from kozlov_lab.io.manifest import expected_dataset_root, load_manifest
from kozlov_lab.io.paths import is_valid_dataset_id
from kozlov_lab.io.runner import execute_run
from .photos import prepare_photo

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = 'object-input-v1'
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_FILE_BYTES = 64 * 1024**2
MAX_OUTPUT_BYTES = 1024**3
PHOTO_SECONDS = 120


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def binding(path: str | Path, limit: int = MAX_TOTAL_BYTES) -> dict:
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('file type or byte limit exceeded')
        digest = hashlib.sha256(); total = 0
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise ValueError('file grew beyond byte limit')
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    if (total != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('file changed while reading')
    return {'path': str(path), 'bytes': total, 'sha256': digest.hexdigest()}


def read_json(path: Path) -> Any:
    with path.open('rb') as stream:
        data = stream.read(4 * 1024**2 + 1)
    if len(data) > 4 * 1024**2:
        raise ValueError('JSON exceeds byte limit')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonfinite JSON constant')
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    # Reject overflowed numeric literals as well as named constants.
    json.dumps(result, allow_nan=False)
    return result


def save(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    pending = path.with_name(path.name + '.pending')
    pending.write_text(data, encoding='utf-8')
    os.replace(pending, path)


def implementation_bindings(project_root: Path) -> list[dict]:
    paths = [project_root/'scripts/prepare_object.py', project_root/'scripts/prepare_video.py',
             project_root/'integrations/video_io/timeline.py']
    for folder in ('integrations/object_input', 'src/kozlov_lab/io', 'src/kozlov_lab/segmentation'):
        paths.extend(sorted((project_root/folder).glob('*.py')))
    return [binding(p) for p in paths]


def _new_output(output: str | Path, project_root: Path) -> Path:
    _no_reparse_ancestors(Path(output).absolute())
    path = Path(output).resolve()
    if path.exists() or (project_root/'outputs').resolve() not in path.parents:
        raise ValueError('output must be new and strictly within project outputs')
    return path


def _no_reparse_ancestors(path: Path) -> None:
    for part in (path,*path.parents):
        if os.path.lexists(part):
            info=part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                raise ValueError('output/dataset reparse ancestors are forbidden')


def _dataset(project_root: Path, dataset_id: str, data_kind: str) -> Path:
    if (not isinstance(dataset_id, str) or not is_valid_dataset_id(dataset_id)
            or len(dataset_id) > 64 or data_kind not in ('real', 'synthetic')):
        raise ValueError('invalid dataset ID or data kind')
    raw=project_root/expected_dataset_root(data_kind, dataset_id)
    _no_reparse_ancestors(raw)
    path = raw.resolve()
    path.relative_to(project_root)
    if path.exists(): raise ValueError('dataset already exists; no overwrite or retry')
    return path


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _enumerate(source: Path) -> tuple[list[Path], list[dict]]:
    if not source.is_dir(): raise ValueError('photo source must be a directory')
    files = []; ignored = []; aliases = set()
    for index, entry in enumerate(source.iterdir()):
        if index >= 1024: raise ValueError('directory has more than 1024 entries')
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('linked/reparse entries are not accepted')
        if not entry.is_file() or entry.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            ignored.append({'name': entry.name, 'reason': 'NON_PHOTO_OR_SUBDIRECTORY_NOT_TRAVERSED'})
            continue
        key = unicodedata.normalize('NFC', entry.name).casefold()
        if key in aliases: raise ValueError('photo filename aliases collide')
        aliases.add(key);files.append(entry.resolve())
    if not 2 <= len(files) <= 96: raise ValueError('photo count must be 2..96; no silent truncation')
    return sorted(files, key=lambda p: unicodedata.normalize('NFC', p.name)), sorted(ignored, key=lambda x:x['name'])


def _rgb(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        if (image.format != 'PNG' or image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1
                or image.getexif().get(274, 1) != 1 or 'transparency' in image.info
                or min(image.size) < 2 or max(image.size) > 1024):
            raise ValueError('canonical image must be bounded opaque RGB PNG')
        image.load(); pixels = image.tobytes();size = image.size
    prefix = f'{size[0]}x{size[1]}:RGB:'.encode('ascii')
    return size, hashlib.sha256(prefix+pixels).hexdigest()


def validate_photo_io(run_file: Path, manifest_file: Path, data_kind: str, project_root: Path, *, max_count: int = 96) -> dict:
    """Rebind T01 outputs and compare every decoded pixel to its derived input."""
    if type(max_count) is not int or max_count not in (96, 192):
        raise ValueError("photo-io max_count must be 96 or 192")
    manifest_load = load_manifest(manifest_file, data_kind)
    if not manifest_load.ok: raise ValueError('invalid photo manifest')
    manifest = manifest_load.manifest
    assert manifest is not None
    count = len(manifest.rows)
    run = read_json(run_file)
    if (not isinstance(run, dict) or run.get('status') != 'SUCCESS'
            or run.get('contract_version') != 'photo-io-v1' or not 2 <= count <= max_count
            or run.get('data_kind') != data_kind or run.get('dataset_id') != manifest.dataset_id
            or run.get('manifest_sha256') != manifest_load.manifest_sha256
            or run.get('implementation_sha256') != source_tree_sha256(project_root)):
        raise ValueError('photo-io success, manifest or implementation mismatch')
    for key, value in [('input_count',count),('ok_count',count),('error_count',0)]:
        if type(run.get(key)) is not int or run[key] != value: raise ValueError('photo-io counts mismatch')
    frames_file = run_file.parent/'frames.jsonl'
    if frames_file.stat().st_size > 4*1024**2: raise ValueError('frames exceeds byte limit')
    rows = [json.loads(line) for line in frames_file.read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != count or any(not isinstance(r,dict) or type(r.get('input_index')) is not int for r in rows):
        raise ValueError('photo-io row type/count mismatch')
    mapping = {r['input_index']:r for r in rows}
    if len(mapping) != count or set(mapping) != set(range(count)): raise ValueError('photo-io index mismatch')
    inputs = [];outputs = [];identities = [];shapes = set();unique = set()
    dataset = (project_root/manifest.dataset_root).resolve()
    _no_reparse_ancestors(project_root/manifest.dataset_root)
    dataset.relative_to(project_root)
    for spec in sorted(manifest.rows, key=lambda r:r.input_index):
        if spec.row_error or spec.mask_canonical is not None: raise ValueError('expected complete derived photo inputs without masks')
        source = (dataset/spec.image_canonical).resolve(strict=True);source.relative_to(dataset)
        src = binding(source, MAX_FILE_BYTES);shape,pixels = _rgb(source)
        row = mapping[spec.input_index]
        expected_output = f'images/{spec.input_index:06d}.png'
        if (row.get('status')!='OK' or row.get('image_output')!=expected_output
                or row.get('image_sha256')!=src['sha256']
                or row.get('frame_id')!=compute_frame_id(manifest.dataset_id,spec.image_canonical,src['sha256'])
                or row.get('image_path')!=spec.image_canonical
                or row.get('mask_output') is not None or row.get('mask_sha256') is not None
                or [row.get('canonical_width'),row.get('canonical_height')]!=list(shape)):
            raise ValueError('photo-io row identity mismatch')
        destination = run_file.parent/expected_output
        _no_reparse_ancestors(destination)
        destination.resolve(strict=True).relative_to(run_file.parent.resolve(strict=True))
        out = binding(destination,MAX_FILE_BYTES)
        if _rgb(destination)!=(shape,pixels): raise ValueError('photo-io changed derived pixels')
        if binding(source,MAX_FILE_BYTES)!=src or binding(destination,MAX_FILE_BYTES)!=out:
            raise ValueError('photo bytes changed during verification')
        inputs.append(src);outputs.append(out);shapes.add(shape);unique.add(pixels)
        identities.append({'input_index':spec.input_index,'frame_id':row['frame_id'],'prepared_dimensions':list(shape),'decoded_rgb_sha256':pixels})
    compatibility = 'READY_FOR_SEPARATE_SFM' if len(unique)>=2 and len(shapes)==1 else 'INSUFFICIENT_UNIQUE_IMAGES' if len(unique)<2 else 'REQUIRES_EQUAL_DIMENSIONS'
    return {'run':binding(run_file),'frames':binding(frames_file),'manifest':binding(manifest_file),
            'prepared_inputs':inputs,'canonical_images':outputs,'identities':identities,
            'count':count,'unique_decoded_images':len(unique),'sfm_compatibility':compatibility}


def _base(output: Path, kind: str, data_kind: str, project_root: Path) -> dict:
    return {'contract_version':CONTRACT,'status':'RUNNING','exit_code':None,'started_utc':utc(),
            'input_kind':kind,'data_kind':data_kind,'implementation':implementation_bindings(project_root),
            'source_bindings':[],'frames':[],'photo_io':None,'output':str(output),
            'runtime':{'python':sys.version.split()[0],'pillow':PIL.__version__,'numpy':np.__version__},
            'segmentation_status':'NOT_RUN','sfm_status':'NOT_RUN','surface_status':'NOT_RUN',
            'CAD_status':'NOT_RUN','coverage_status':'NOT_EVALUATED','scale_status':'unknown','length_unit':None}


def _verify_bindings(rows: list[dict]) -> None:
    for row in rows:
        if binding(row['path']) != row: raise ValueError('file binding drift: '+row['path'])


def _finish(output: Path, record: dict, start: float) -> dict:
    record.update(ended_utc=utc(),wall_seconds=time.monotonic()-start)
    save(output/'RESULT.json',record)
    return record


def prepare_photos(source_dir: str | Path, dataset_id: str, data_kind: str,
                   output: str | Path, *, project_root: Path = ROOT) -> dict:
    project_root=Path(project_root).resolve();source=Path(source_dir).resolve(strict=True)
    output=_new_output(output,project_root);dataset=_dataset(project_root,dataset_id,data_kind)
    if any(_overlap(a,b) for a,b in ((source,output),(source,dataset),(dataset,output))):
        raise ValueError('source, dataset and output must be disjoint')
    files,ignored=_enumerate(source)
    sources=[binding(p,MAX_FILE_BYTES) for p in files]
    if sum(r['bytes'] for r in sources)>MAX_TOTAL_BYTES: raise ValueError('photo source bytes exceed 512 MiB')
    start=time.monotonic();deadline=start+PHOTO_SECONDS
    record=_base(output,'photos',data_kind,project_root)
    record.update(dataset_id=dataset_id,dataset_root=str(dataset),source_directory=str(source),
                  source_bindings=sources,ignored_entries=ignored,
                  limits={'photo_seconds_soft':PHOTO_SECONDS,'source_bytes':MAX_TOTAL_BYTES,'output_bytes_soft':MAX_OUTPUT_BYTES},
                  frames=[{'source_index':i,'source':s,'status':'NOT_RUN'} for i,s in enumerate(sources)])
    output.mkdir(parents=True,exist_ok=False)
    try:
        save(output/'RESULT.json',record);(dataset/'frames').mkdir(parents=True,exist_ok=False)
        generated=0
        for i,row in enumerate(record['frames']):
            if time.monotonic()>=deadline: raise TimeoutError('shared photo preparation budget exceeded')
            row['status']='PROCESSING';save(output/'RESULT.json',record)
            with Path(row['source']['path']).open('rb') as stream:data=stream.read(MAX_FILE_BYTES+1)
            if len(data)>MAX_FILE_BYTES or hashlib.sha256(data).hexdigest()!=row['source']['sha256']:
                raise ValueError('source changed before decode')
            png,metadata=prepare_photo(data)
            if generated+len(png)>MAX_OUTPUT_BYTES: raise TimeoutError('prepared output budget exceeded')
            dest=dataset/'frames'/f'{i:06d}.png'
            with dest.open('xb') as stream:stream.write(png)
            generated+=len(png);row.update(status='PREPARED',prepared=binding(dest),metadata=metadata)
            save(output/'RESULT.json',record)
        manifest={'contract_version':'photo-io-v1','dataset_id':dataset_id,'data_kind':data_kind,
                  'dataset_root':str(dataset.relative_to(project_root)).replace('\\','/'),
                  'frames':[{'image':f'frames/{i:06d}.png','mask':None} for i in range(len(files))]}
        manifest_file=output/'photo_manifest.json';save(manifest_file,manifest)
        if time.monotonic()>=deadline: raise TimeoutError('budget exhausted before photo-io')
        photo=execute_run(load_manifest(manifest_file,data_kind),output_base=output/'photo_io',data_kind=data_kind,project_root=project_root)
        record['photo_io_attempt']={'run':str(photo.run_dir/'run.json'),'exit_code':photo.exit_code,'status':photo.status}
        if photo.exit_code!=0: raise ValueError('photo-io did not accept every prepared input')
        record['photo_io']=validate_photo_io(photo.run_dir/'run.json',manifest_file,data_kind,project_root)
        _verify_bindings(sources+record['implementation'])
        if time.monotonic()>=deadline: raise TimeoutError('photo preparation budget exceeded after verification')
        bytes_now=sum(p.stat().st_size for d in (output,dataset) for p in d.rglob('*') if p.is_file())
        record['output_bytes_observed']=bytes_now
        if bytes_now>MAX_OUTPUT_BYTES: raise TimeoutError('total prepared output budget exceeded')
        record.update(status='INPUTS_PREPARED',exit_code=0)
    except Exception as exc:
        for row in record['frames']:
            if row['status']=='PROCESSING':row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        code=5 if isinstance(exc,TimeoutError) else 2 if isinstance(exc,(ValueError,FrameIssue)) else 4
        record.update(status='RESOURCE_LIMIT' if code==5 else 'INPUT_ERROR' if code==2 else 'BACKEND_ERROR',exit_code=code,error=f'{type(exc).__name__}: {exc}')
    return _finish(output,record,start)


def _validated_video(directory: Path, project_root: Path) -> tuple[dict,dict,list[dict]]:
    result_file=directory/'RESULT.json';v=read_json(result_file)
    if (not isinstance(v,dict) or v.get('contract_version')!='video-photo-v1'
            or v.get('status')!='SUCCESS' or type(v.get('exit_code')) is not int or v['exit_code']!=0
            or v.get('data_kind') not in ('real','synthetic')):
        raise ValueError('video stage is not a successful video-photo-v1 run')
    source=binding(v['source_video'])
    if source['sha256']!=v.get('source_sha256') or source['sha256']!=v.get('source_sha256_after'):
        raise ValueError('video source drift')
    timeline=read_json(directory/'timeline.json')
    if not isinstance(timeline,dict) or {k:x for k,x in timeline.items() if k!='frames'}!=v.get('timeline'):
        raise ValueError('video timeline metadata mismatch')
    all_frames=timeline.get('frames')
    if (not isinstance(all_frames,list) or not 2<=len(all_frames)<=20000
            or type(v.get('decoded_count')) is not int or v['decoded_count']!=len(all_frames)):
        raise ValueError('video timeline count mismatch')
    for i,frame in enumerate(all_frames):
        if (not isinstance(frame,dict) or type(frame.get('decode_index')) is not int or frame['decode_index']!=i
                or type(frame.get('pts')) is not int or frame.get('time_base')!=timeline.get('time_base')
                or (i>0 and frame['pts']<=all_frames[i-1]['pts'])):
            raise ValueError('video timeline index/PTS mismatch')
    from integrations.video_io.timeline import select_uniform, select_all
    selection=v.get('selection',{})
    requested=selection.get('requested')
    whole=selection.get('method')=='all_decoded_frames'
    if whole:
        if (v.get('profile')!='whole-video-192-v1' or selection.get('profile')!='whole-video-192-v1'
                or type(requested) is not int or requested!=len(all_frames)):
            raise ValueError('whole-video profile/count mismatch')
        selected=select_all(timeline)
    else:
        if selection.get('method')!='uniform_presentation_time_nearest':raise ValueError('unknown video selection method')
        if type(requested) is not int or not 2<=requested<=96:raise ValueError('video selection count mismatch')
        selected=select_uniform(timeline,requested)
    if not isinstance(v.get('frames'),list) or len(v['frames'])!=len(selected):raise ValueError('video selected count mismatch')
    for row,original in zip(v['frames'],selected):
        if (any(row.get(k)!=original[k] for k in ('decode_index','pts','time_base'))
                or row.get('export_pts_verified') is not True):
            raise ValueError('video selected PTS does not match timeline')
    expected={'scripts/prepare_video.py','integrations/video_io/timeline.py'}
    if set(v.get('implementation',{}))!=expected: raise ValueError('video implementation set mismatch')
    bound=[source,binding(result_file),binding(directory/'timeline.json')]
    for name,digest in v['implementation'].items():
        b=binding(project_root/name)
        if b['sha256']!=digest: raise ValueError('video implementation drift')
        bound.append(b)
    if not isinstance(v.get('tools'),dict) or len(v['tools'])!=2:raise ValueError('video tool set mismatch')
    for name,digest in v['tools'].items():
        b=binding(name)
        if b['sha256']!=digest: raise ValueError('video tool drift')
        bound.append(b)
    t01=v['photo_io'];run_file=Path(t01['run']).resolve(strict=True);run_file.relative_to(directory)
    if binding(run_file)['sha256']!=t01['sha256'] or binding(run_file.parent/'frames.jsonl')['sha256']!=t01['frames_sha256']:
        raise ValueError('video photo-io drift')
    checked=validate_photo_io(run_file,directory/'photo_manifest.json',v['data_kind'],project_root,max_count=192 if whole else 96)
    manifest=read_json(directory/'photo_manifest.json')
    if v['dataset_root']!=manifest['dataset_root']:raise ValueError('video dataset root mismatch')
    accepted=[r for r in v['frames'] if r.get('photo_input_index') is not None]
    if len(accepted)!=checked['count'] or {r['photo_input_index'] for r in accepted}!=set(range(checked['count'])):
        raise ValueError('video accepted identity map mismatch')
    by_index={r['input_index']:r for r in checked['identities']}
    for row in accepted:
        if type(row['photo_input_index']) is not int or row.get('status')!='PHOTO_IO_OK' or row.get('photo_frame_id')!=by_index[row['photo_input_index']]['frame_id']:
            raise ValueError('video frame identity mismatch')
        actual=binding(project_root/v['dataset_root']/row['image'],MAX_FILE_BYTES)
        if actual['sha256']!=row.get('sha256'):raise ValueError('video extracted frame drift')
        identity=by_index[row['photo_input_index']]
        if (actual!=checked['prepared_inputs'][row['photo_input_index']]
                or [row.get('width'),row.get('height')]!=identity['prepared_dimensions']
                or row.get('decoded_rgb_sha256')!=identity['decoded_rgb_sha256']):
            raise ValueError('video source frame and photo index do not refer to the same pixels')
    return v,checked,bound


def adopt_video(video_run: str | Path, output: str | Path, *, project_root: Path = ROOT) -> dict:
    """Read-only adoption of existing input results; no decode/reconstruction/retry."""
    project_root=Path(project_root).resolve();directory=Path(video_run).resolve(strict=True)
    output=_new_output(output,project_root)
    if _overlap(output,directory):raise ValueError('adoption output must be separate from video run')
    start=time.monotonic();video,checked,bound=_validated_video(directory,project_root)
    record=_base(output,'video',video['data_kind'],project_root)
    record.update(status='INPUTS_PREPARED',exit_code=0,reused_video_run=str(directory),
                  source_bindings=bound,photo_io=checked,frames=video['frames'],
                  video_timeline_verification='BOUND_EXISTING_REPORT_NOT_DECODE_REPLAY',
                  native_processes_started=0)
    _verify_bindings(bound+record['implementation']);output.mkdir(parents=True,exist_ok=False)
    return _finish(output,record,start)


def prepare_video(video: str | Path, dataset_id: str, data_kind: str, output: str | Path,
                  *, count: int = 48, all_frames: bool = False, ffmpeg: str | None = None, ffprobe: str | None = None,
                  project_root: Path = ROOT) -> dict:
    """Invoke unchanged bounded video preparation; never invokes SAM3/SfM/MVS."""
    project_root=Path(project_root).resolve()
    if project_root!=ROOT:raise ValueError('native video wrapper requires the actual project root')
    output=_new_output(output,project_root);dataset=_dataset(project_root,dataset_id,data_kind)
    source=Path(video).resolve(strict=True)
    if _overlap(output,dataset) or output in source.parents:raise ValueError('video input/output overlap')
    start=time.monotonic();record=_base(output,'video',data_kind,project_root)
    record['source_bindings']=[binding(source)];output.mkdir(parents=True,exist_ok=False);save(output/'RESULT.json',record)
    try:
        path=project_root/'scripts/prepare_video.py';spec=importlib.util.spec_from_file_location('_object_video_entry',path)
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        args=SimpleNamespace(video=str(source),dataset_id=dataset_id,data_kind=data_kind,output=str(output/'video'),
                             count=count,ffmpeg=ffmpeg,ffprobe=ffprobe,stream_index=None,min_sharpness=0.0)
        if type(all_frames) is not bool:raise ValueError('all_frames must be boolean')
        if all_frames:args.all_frames=True
        code=module.execute(args);record['video_exit_code']=code
        if type(code) is not int or code!=0:raise ValueError('video stage failed; inspect retained video/RESULT.json')
        v,checked,bound=_validated_video(output/'video',project_root)
        _verify_bindings(record['implementation'])
        record.update(status='INPUTS_PREPARED',exit_code=0,photo_io=checked,source_bindings=bound,
                      frames=v['frames'],reused_video_run=str(output/'video'),
                      video_timeline_verification='BOUND_EXISTING_REPORT_NOT_DECODE_REPLAY')
    except Exception as exc:
        code=record.get('video_exit_code')
        code=code if type(code) is int and code in (2,4,5) else 2 if isinstance(exc,(ValueError,OSError)) else 4
        record.update(status='PREPARATION_FAILED',exit_code=code,error=f'{type(exc).__name__}: {exc}')
    return _finish(output,record,start)


def verify_input(output: str | Path, *, project_root: Path = ROOT) -> dict:
    """Validate prepared artifacts; does not re-run or promote downstream stages."""
    project_root=Path(project_root).resolve();output=Path(output).resolve(strict=True)
    r=read_json(output/'RESULT.json')
    if (not isinstance(r,dict) or r.get('contract_version')!=CONTRACT or r.get('status')!='INPUTS_PREPARED'
            or type(r.get('exit_code')) is not int or r['exit_code']!=0):raise ValueError('input receipt is not successful')
    if r.get('output')!=str(output):raise ValueError('receipt output identity mismatch')
    if r.get('runtime')!={'python':sys.version.split()[0],'pillow':PIL.__version__,'numpy':np.__version__}:
        raise ValueError('input preparation runtime changed')
    if r.get('implementation')!=implementation_bindings(project_root):raise ValueError('input implementation drift')
    _verify_bindings(r['source_bindings'])
    if r['input_kind']=='video':
        video,checked,bound=_validated_video(Path(r['reused_video_run']),project_root)
        if r['data_kind']!=video['data_kind'] or r['source_bindings']!=bound or r['frames']!=video['frames']:
            raise ValueError('video adoption record differs')
    elif r['input_kind']=='photos':
        deadline=time.monotonic()+PHOTO_SECONDS
        if not 2<=len(r['frames'])<=96 or len(r['frames'])!=len(r['source_bindings']):raise ValueError('photo receipt counts invalid')
        for i,row in enumerate(r['frames']):
            if time.monotonic()>=deadline:raise ValueError('readonly photo revalidation budget exceeded')
            if row.get('status')!='PREPARED' or type(row.get('source_index')) is not int or row['source_index']!=i or row['source']!=r['source_bindings'][i]:
                raise ValueError('photo receipt row mismatch')
            expected=Path(r['dataset_root'])/'frames'/f'{i:06d}.png'
            if row['prepared']!=binding(expected,MAX_FILE_BYTES):raise ValueError('prepared photo drift')
            dimensions,pixels=_rgb(expected)
            if (row['metadata'].get('prepared_dimensions')!=list(dimensions)
                    or row['metadata'].get('decoded_rgb_sha256')!=pixels
                    or row['metadata'].get('output_sha256')!=row['prepared']['sha256']):raise ValueError('prepared metadata mismatch')
            with Path(row['source']['path']).open('rb') as stream:data=stream.read(MAX_FILE_BYTES+1)
            if len(data)>MAX_FILE_BYTES or hashlib.sha256(data).hexdigest()!=row['source']['sha256']:
                raise ValueError('source changed before readonly decode')
            png,metadata=prepare_photo(data)
            if metadata!=row['metadata'] or hashlib.sha256(png).hexdigest()!=row['prepared']['sha256']:
                raise ValueError('source-to-prepared transformation mismatch')
        checked=validate_photo_io(Path(r['photo_io']['run']['path']),output/'photo_manifest.json',r['data_kind'],project_root)
        if len(r['frames'])!=checked['count'] or [x['prepared'] for x in r['frames']]!=checked['prepared_inputs']:
            raise ValueError('prepared dataset and T01 linkage mismatch')
    else:raise ValueError('unknown input kind')
    if checked!=r['photo_io']:raise ValueError('photo-io receipt drift')
    _verify_bindings(r['source_bindings'])
    return {'status':'INPUT_BINDINGS_VALID','data_kind':r['data_kind'],'input_kind':r['input_kind'],
            'count':checked['count'],'photo_io_run':str(Path(checked['run']['path']).parent),
            'sfm_compatibility':checked['sfm_compatibility'],'native_processes_started':0,
            'segmentation_status':'NOT_RUN','surface_status':'NOT_RUN','CAD_status':'NOT_RUN'}
