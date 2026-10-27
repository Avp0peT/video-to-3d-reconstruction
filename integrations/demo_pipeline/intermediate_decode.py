"""Bounded extraction of original frames in the existing canonical video domain."""
from fractions import Fraction
from pathlib import Path
import re
import time
import json
import numpy as np
from PIL import Image
from scripts.prepare_video import bounded_run, inspect_image, expected_orientation, LimitError
from integrations.video_io.timeline import parse_showinfo, validate_export
from .temporal_review import bind, read_bound, receipt_path, same_path

SUFFIX = (r",scale=w=min(1024\,iw):h=min(1024\,ih):force_original_aspect_ratio=decrease,"
          "format=rgb24,showinfo")


def _recipe(argv, source, stream):
    """Accept exactly the known extraction recipe; reject silent transformations."""
    expected = ['-hide_banner','-nostdin','-n','-loglevel','info','-copyts','-xerror',
                '-threads','2','-protocol_whitelist','file','-format_whitelist','mov,matroska,avi',
                '-err_detect','explode','-i',str(source),'-map',f'0:{stream}',
                '-an','-sn','-dn','-filter_threads','2','-vf',None,
                '-fps_mode','passthrough','-enc_time_base','filter','-avoid_negative_ts','disabled',
                '-c:v','png','-threads','2','-start_number','0','-f','image2',None]
    if len(argv) != len(expected)+1:
        raise ValueError('unsupported extraction recipe')
    for actual, wanted in zip(argv[1:], expected):
        if wanted is not None and actual != wanted:
            raise ValueError('unsupported extraction recipe')
    vf = argv[argv.index('-vf')+1]
    if not vf.endswith(SUFFIX) or not re.fullmatch(r"select='(?:1|eq\(n,\d+\)(?:\+eq\(n,\d+\))*)'", vf[:-len(SUFFIX)]):
        raise ValueError('unsupported extraction filter recipe')
    return argv.index('-vf')+1


def decode_intermediates(discovery, frames, decode_indices, output, refs, *, deadline, budget=lambda: None):
    indices = list(decode_indices)
    if not 1 <= len(indices) <= 64 or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('1..64 unique nonnegative integer decode indices required')
    indices.sort()
    budget()
    if time.monotonic() >= deadline:
        raise TimeoutError('extraction deadline exhausted')
    receipt = read_bound(receipt_path(None, discovery), refs)
    if receipt.get('contract_version') != 'object-input-v1' or receipt.get('status') != 'INPUTS_PREPARED' or receipt.get('input_kind') != 'video':
        raise ValueError('prepared video object-input receipt required')
    root = Path(receipt['reused_video_run']).resolve(strict=True)
    declared = {str(Path(r['path']).resolve()): r for r in receipt['source_bindings']}
    vp = root/'RESULT.json'
    if str(vp) not in declared:
        raise ValueError('unbound video receipt')
    video = read_bound(vp, refs, declared[str(vp)])
    if video.get('contract_version') != 'video-photo-v1' or video.get('status') != 'SUCCESS':
        raise ValueError('successful video receipt required')
    source = Path(video['source_video']).resolve(strict=True)
    if not same_path(source, discovery['source']):
        raise ValueError('discovery source differs')
    source_ref = bind(source, refs, {'sha256': video['source_sha256']})
    if source_ref['sha256'] != video['source_sha256_after']:
        raise ValueError('source after binding differs')
    timeline = read_bound(root/'timeline.json', refs)
    if len(timeline['frames']) != video['decoded_count'] or indices[-1] >= len(timeline['frames']):
        raise ValueError('decode index outside timeline')
    if {k:v for k,v in timeline.items() if k != 'frames'} != video['timeline']:
        raise ValueError('video timeline metadata differs')
    selected = [timeline['frames'][i] for i in indices]
    if any(r['decode_index'] != i for i,r in zip(indices,selected)):
        raise ValueError('timeline decode identity differs')
    old = read_bound(root/'extract.process.json', refs)
    argv = list(old['argv'])
    vf_index = _recipe(argv, source, timeline['stream_index'])
    if old.get('returncode') != 0:
        raise ValueError('original extraction did not complete')
    tool = Path(argv[0]).resolve(strict=True)
    if str(tool) not in video['tools']:
        raise ValueError('extraction tool absent from video binding')
    tool_ref = bind(tool, refs, {'sha256':video['tools'][str(tool)]})
    endpoints = {f['decode_index']: f for f in frames if f['decode_index'] in indices}
    ledger = {f['decode_index']:f for f in video['frames'] if f['status']=='PHOTO_IO_OK'}
    if not endpoints:
        raise ValueError('at least one canonical endpoint required')
    for index, frame in endpoints.items():
        if index not in ledger or frame['pts'] != ledger[index]['pts'] or frame['time_base'] != ledger[index]['time_base']:
            raise ValueError('endpoint video identity differs')
        if ledger[index].get('export_pts_verified') is not True:
            raise ValueError('endpoint video export timestamp was not verified')
        bind(frame['path'],refs,{'sha256':frame['sha256']})
        canonical = inspect_image(frame['path'])
        if (canonical['width'],canonical['height']) != (ledger[index]['width'],ledger[index]['height']):
            raise ValueError('endpoint canonical dimensions differ from video ledger')
        if canonical['decoded_rgb_sha256'] != ledger[index]['decoded_rgb_sha256']:
            raise ValueError('endpoint canonical RGB differs from video ledger')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = output/'images'; images.mkdir()
    logs = output/'native'; logs.mkdir()
    argv[vf_index] = "select='" + '+'.join(f'eq(n,{i})' for i in indices) + "'" + SUFFIX
    argv[-1] = str(images/'frame_%06d.png')
    plan_path = output/'EXTRACTION_PLAN.json'
    plan_path.write_text(json.dumps(dict(schema='original-intermediate-extraction-plan-v1',
        source_video=source_ref,tool=tool_ref,argv=argv,decode_indices=indices,selected_frames=selected,
        endpoint_indices=sorted(endpoints),retry_count=0),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plan_ref = bind(plan_path,refs)
    budget()
    try:
        _, log = bounded_run(argv, logs, 'extract', deadline, images, old.get('env_overrides',{}))
    except LimitError as error:
        message = str(error)
        if 'deadline' in message:
            raise TimeoutError(message) from error
        if 'logs exceeded' in message or 'output size exceeded' in message:
            raise MemoryError(message) from error
        raise
    budget()
    observed = parse_showinfo(log.read_text(encoding='utf-8', errors='replace'))
    validate_export(timeline, selected, observed)
    paths = [images/f'frame_{i:06d}.png' for i in range(len(indices))]
    if sorted(images.iterdir()) != paths:
        raise ValueError('extracted image set differs')
    mapping = {}; rows = []
    for index, row, shown, path in zip(indices, selected, observed, paths):
        budget(); info = inspect_image(path)
        if (info['width'],info['height']) != (shown['width'],shown['height']) or not expected_orientation(timeline,info['width'],info['height']):
            raise ValueError('extracted dimensions differ')
        verified = index in endpoints
        if verified and info['decoded_rgb_sha256'] != ledger[index]['decoded_rgb_sha256']:
            raise ValueError('redecoded endpoint RGB differs')
        with Image.open(path) as image:
            gray = np.array(image.convert('L'))
        item = dict(decode_index=index,path=str(path),width=info['width'],height=info['height'],
                    pts=row['pts'],time_base=row['time_base'],
                    time_seconds=float(Fraction(row['pts'])*Fraction(*row['time_base'])),
                    file_sha256=info['sha256'],sha256=info['sha256'],rgb_sha256=info['decoded_rgb_sha256'],
                    original_mask_status='UNKNOWN_NOT_SEGMENTED',endpoint_verified=verified)
        rows.append(item.copy()); mapping[index]=dict(item,gray=gray,mask=None)
        bind(path,refs,{'sha256':info['sha256']})
    bind(source,refs,source_ref); bind(tool,refs,tool_ref)
    for frame in endpoints.values():
        bind(frame['path'],refs,{'sha256':frame['sha256']})
    evidence = [bind(logs/'extract.process.json',refs), bind(log,refs)]
    return mapping, dict(schema='original-intermediate-extraction-v1',status='EXTRACTED_RGB_VERIFIED',
                         source_video=source_ref,tool=tool_ref,decode_indices=indices,frames=rows,
                         verified_endpoint_indices=sorted(endpoints),extraction_plan=plan_ref,native_evidence=evidence,
                         source_normalization_caveat='Uses the already-bound video unchanged; any prior SAR declaration remains an assumption, not calibration.',
                         geometry_accepted=False,new_segmentation=False)
