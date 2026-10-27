"""Read-only joins of unaccepted boundary evidence to existing native cameras.

No pose interpolation, label promotion or historical code-hash admission gate.
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from integrations.viewer.ply_data import read_ply
from integrations.mvs_surface.scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap
from integrations.mvs_masks.mask_geometry import warp_binary_mask
from .face_mask_adapter import verify_depth
from .face_mask_adapter import _read as _strict_read

_HASH_CACHE = {}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _bind(path, refs, expected=None):
    p = Path(path).resolve(strict=True)
    _need(p.is_file(), 'regular input file required')
    stat = p.stat(); stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    cached = _HASH_CACHE.get(str(p))
    if cached is not None and cached[0] == stamp:
        digest = cached[1]
    else:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        after = p.stat()
        _need(stamp == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino), 'input changed while hashing')
        digest = h.hexdigest(); _HASH_CACHE[str(p)] = (stamp, digest)
    row = dict(path=str(p), bytes=stat.st_size, sha256=digest)
    if expected is not None:
        if isinstance(expected, str):
            _need(row['sha256'] == expected, 'input hash mismatch: ' + str(p))
        else:
            _need(all(row[k] == expected[k] for k in ('bytes', 'sha256') if k in expected),
                  'input binding mismatch: ' + str(p))
    previous = [r for r in refs if r['path'] == row['path']]
    _need(not previous or all(r == row for r in previous), 'input changed since binding')
    if not previous:
        refs.append(row)
    return p


def _read(path, refs, expected=None):
    return _strict_read(_bind(path, refs, expected))


def _image(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def _binary(path, shape=None):
    with Image.open(path) as im:
        a = np.array(im)
        _need(im.mode == 'L' and np.isin(a, [0, 255]).all(), 'binary L image required')
    _need(shape is None or a.shape == shape, 'binary dimensions differ')
    return a


def _camera_check(frame, image, exported):
    _need(frame['target_image_id'] == image['global_id'] and
          frame['exported_name'] == Path(image['name']).name == exported['exported_name'],
          'global image identity mismatch')
    t = frame['target_camera']; s = frame['source_camera']; pose = exported['source_pose']
    _need(t['model'] == 'PINHOLE' and (t['width'], t['height']) == (image['width'], image['height']),
          'target camera domain mismatch')
    fx, fy, cx, cy = t['params']
    k = np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])
    _need(np.allclose(k, image['K'], rtol=0, atol=1e-7), 'COLMAP/native centre convention mismatch')
    _need(s['model'] == exported['source_camera'].get('model', exported['source_camera'].get('model_name')) and
          (s['width'], s['height']) == (exported['source_camera']['width'], exported['source_camera']['height']) and
          np.array_equal(s['params'], exported['source_camera']['params']), 'source camera mismatch')
    rotation = np.asarray(pose['rotation_matrix']); centre = -rotation.T @ np.asarray(pose['translation'])
    _need(np.allclose(rotation, image['R'], rtol=0, atol=1e-7) and
          np.allclose(centre, image['C'], rtol=0, atol=1e-7), 'source/native pose mismatch')


def _prove_derivation(view, union, refs):
    """Exact historical extraction/ordinal chain, never an RGB tolerance."""
    start = len(refs); e = view['frame']; i = view['decode_index']
    up = _read(union/'PLAN.json', refs)
    vrpath = Path(up['video_result_path']); vr = _read(vrpath, refs, up['video_result_sha256'])
    timeline = _read(up['timeline_path'], refs, up['timeline_sha256'])
    _need(vr['status'] == 'SUCCESS' and vr['source_sha256'] == vr['source_sha256_after'] == up['source_video_sha256'] and
          Path(vr['source_video']).resolve() == Path(up['source_video_path']).resolve(), 'video source identity mismatch')
    _bind(vr['source_video'], refs, vr['source_sha256'])
    _need(timeline['rotation_degrees'] == 0 and timeline['display_matrix'] is None and
          timeline['sample_aspect_ratio'] == '1:1' and timeline['stream_index'] == 0, 'unsupported source axes')
    selected = [x for x in vr['frames'] if x['decode_index'] == i]
    _need(len(selected) == 1, 'UNVERIFIED_FRAME_IDENTITY: not a bound extraction endpoint')
    f = selected[0]; t = [x for x in timeline['frames'] if x['decode_index'] == i]
    _need(len(t) == 1 and f['pts'] == t[0]['pts'] and f['time_base'] == t[0]['time_base'] and
          f['export_pts_verified'] is True and f['status'] == 'PHOTO_IO_OK', 'ordinal/PTS mismatch')
    io = vr['photo_io']; runpath = _bind(io['run'], refs, io['sha256'])
    fp = _bind(runpath.parent/'frames.jsonl', refs, io['frames_sha256'])
    # Apply the same duplicate-key/nonfinite JSON rejection to JSONL records.
    import json
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    entries = [json.loads(line, object_pairs_hook=pairs, parse_constant=bad_constant)
               for line in fp.read_text(encoding='utf-8').splitlines() if line]
    entries = [x for x in entries if x['input_index'] == f['photo_input_index']]
    _need(len(entries) == 1, 'photo input identity mismatch'); row = entries[0]
    canonical = _bind(runpath.parent/row['image_output'], refs, e['canonical_image_sha256'])
    _need(canonical == Path(e['canonical_image_path']).resolve() and row['frame_id'] == f['photo_frame_id'] and
          row['status'] == 'OK' and row['orientation_applied'] == 1 and row['exif_orientation_original'] in (None,1) and
          np.array_equal(row['raw_to_canonical'], np.eye(3)) and
          (row['raw_width'], row['raw_height']) == (row['canonical_width'], row['canonical_height']) == (f['width'], f['height']) == (up['width'],up['height']), 'photo transform mismatch')
    raw = _bind(Path(vr['dataset_root'])/f['image'], refs, f['sha256'])
    raw_rgb = _image(raw)
    pixel_header = f"{raw_rgb.shape[1]}x{raw_rgb.shape[0]}:RGB:".encode('ascii')
    _need(hashlib.sha256(pixel_header + raw_rgb.tobytes()).hexdigest() == f['decoded_rgb_sha256'] and
          np.array_equal(_image(raw), _image(canonical)) and
          np.array_equal(_image(canonical), _image(e['source_image_path'])), 'endpoint pixels changed')
    extraction = _read(vrpath.parent/'extract.process.json', refs); argv = extraction['argv']
    _need(extraction['returncode'] == 0 and argv[argv.index('-i')+1] == vr['source_video'] and
          argv[argv.index('-map')+1] == '0:0' and argv[argv.index('-fps_mode')+1] == 'passthrough', 'extract process mismatch')
    expected_indices = [x['decode_index'] for x in sorted(vr['frames'], key=lambda x:x['output_index'])]
    selector = "select='" + '+'.join(f'eq(n,{j})' for j in expected_indices) + "'"
    vf = selector + ',scale=w=min(1024\\,iw):h=min(1024\\,ih):force_original_aspect_ratio=decrease,format=rgb24,showinfo'
    _need(argv[argv.index('-vf')+1] == vf and f['output_index'] == expected_indices.index(i), 'unproved extract filter or ordinal')
    _need(timeline['width']*up['height'] == timeline['height']*up['width'], 'nonproportional canonical domain')
    return dict(status='VERIFIED_BOUND_ORDINAL_PTS_RESAMPLING_CHAIN', decode_index=i, pts=f['pts'],
                time_base=f['time_base'], source_video_sha256=up['source_video_sha256'],
                canonical_size=[up['width'],up['height']], native_size=[timeline['width'],timeline['height']],
                resampling_diff_known=True, bindings=refs[start:])


def prepare_context(surface_run, local_run):
    """Return bound scene, decode-index frames, views and candidate mesh arrays."""
    source = Path(surface_run).resolve(strict=True); local = Path(local_run).resolve(strict=True); refs = []
    lr = _read(local/'RESULT.json', refs)
    _need(lr.get('schema') == 'local-boundary-surface-v1' and
          lr.get('status') == 'LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED' and
          lr.get('source_unchanged') is True and lr.get('geometry_accepted') is False and
          Path(lr['source_run']).resolve() == source, 'local source contract mismatch')
    nr = _read(source/'surface/result.json', refs)
    _need(nr.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and not nr.get('input_drift'), 'native surface refused')
    masks = _read(source/'masks/RESULT.json', refs); export = _read(source/'export/EXPORT.json', refs)
    _need(masks.get('schema') == 'region-surface-masks-v1' and masks.get('coverage_status') == 'COMPLETE' and
          masks.get('adapter_source_chain') == 'VERIFIED' and not masks.get('input_drift') and not masks.get('output_drift'), 'mask contract refused')
    _need(export.get('schema') == 'region-surface-export-v1' and export.get('source_unchanged') is True,
          'export contract refused')
    fmap = _read(source/'export/FRAME_MAP.json', refs, export['frame_map_sha256'])
    _need(fmap.get('schema') == 'region-surface-frame-map-v1', 'frame map schema mismatch')
    scene = read_scene(_bind(source/'bridge/scene.mvs', refs, masks['scene_binding']))
    _need(np.array_equal(scene['transform'], np.eye(4)), 'nonidentity scene transform')
    images = {Path(x['name']).name: x for x in scene['images']}
    fm = {x['exported_name']: x for x in fmap['frames']}; mf = {x['exported_name']: x for x in masks['frames']}
    _need(0 < len(images) <= 256 and len(images) == len(scene['images']) == len(fm) == len(fmap['frames']) == len(mf) == len(masks['frames']) and images.keys() == fm.keys() == mf.keys(), 'view set mismatch')
    _need(len({x['global_id'] for x in images.values()}) == len(images), 'duplicate global image ID')
    record = lr['outputs']['retained']; path = Path(record['path']).resolve()
    _need(path == local/'retained.ply', 'retained output escaped local run')
    mesh = read_ply(_bind(path, refs, record), max_bytes=16<<20, max_vertices=200000, max_faces=400000)
    _need(len(mesh['vertices']) == record['vertices'] and len(mesh['faces']) == record['faces'], 'mesh counts mismatch')
    _bind(local/'retained.mapping.npz', refs); _bind(local/'LINEAGE.npz', refs)
    frames = {}; views = []; roots = set()
    for name, image in images.items():
        f, e = mf[name], fm[name]; i = f['decode_index']
        _need(type(i) is int and i not in frames and i == e['decode_index'] and f['input_index'] == e['input_index'] and f['status'] == 'OK' and f['readback_verified'] is True, 'frame identity mismatch')
        _camera_check(f, image, e)
        _need(f['source_mask_path'] == e['source_mask_path'] and f['source_mask_sha256'] == e['source_mask_sha256'], 'source mask identity mismatch')
        _bind(f['source_mask_path'], refs, f['source_mask_sha256'])
        for key in ('mask', 'validity'):
            _binary(_bind(f[key]['path'], refs, f[key]), (image['height'], image['width']))
        _bind(e['source_image_path'], refs, e['source_image_sha256'])
        _bind(e['exported_png_path'], refs, e['exported_png_sha256'])
        _need(np.array_equal(_image(e['source_image_path']), _image(e['exported_png_path'])), 'export RGB pixels changed')
        roots.add(Path(f.get('mask_source_root', Path(f['source_mask_path']).parent.parent)).resolve())
        view = dict(decode_index=i, camera_scene=image, frame=e, mask_record=f, depth_path=None,
                    surface_run=str(source), local_run=str(local))
        frames[i] = view; views.append(view)
    _need(len(roots) == 1, 'multiple source mask roots')
    union = roots.pop()
    if not (union/'RESULT.json').is_file():
        return dict(status='NOT_APPLICABLE', reason='NO_BOUND_UNION_RESULT', union_run=None,
                    scene=scene, frames=frames, views=views, source_bindings=refs, mesh=mesh)
    union_result = _read(union/'RESULT.json', refs)
    literal = ((union_result.get('status') == 'FULL_TIMELINE_FUSED_EXPORTED' and union_result.get('mode') == 'union') or
               (union_result.get('status') == 'FULL_TIMELINE_EXPORTED' and union_result.get('fusion', {}).get('mode') == 'union'))
    if not literal:
        return dict(status='NOT_APPLICABLE', reason='BOUND_LITERAL_UNION_REQUIRED', union_run=None,
                    scene=scene, frames=frames, views=views, source_bindings=refs, mesh=mesh)
    assembly = _read(union/'ASSEMBLY.json', refs); ar = {x['decode_index']: x for x in assembly['rows']}
    for i, view in frames.items():
        f = view['mask_record']; a = ar[i]
        _bind(a['mask_path'], refs, a['mask_sha256'])
        _need(a['mask_sha256'] == f['source_mask_sha256'], 'union frame binding mismatch')
        view['union_run'] = str(union)
        try:
            view['frame_identity'] = _prove_derivation(view, union, refs)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            view['frame_identity'] = dict(status='UNVERIFIED_FRAME_IDENTITY', reason=str(exc))
    done = set()
    for path in sorted((source/'surface/work').glob('depth*.dmap')):
        d = read_dmap(path); name = Path(d.image_name_utf8.replace('\\', '/')).name
        _need(name in images and name not in done, 'unknown/duplicate depth image')
        verify_depth(d, images[name]); _bind(path, refs, dict(sha256=d.sha256, bytes=d.file_bytes))
        frames[mf[name]['decode_index']]['depth_path'] = str(path); done.add(name)
    return dict(status='BOUND_CONTEXT_NOT_ACCEPTED', scene=scene, frames=frames, views=views,
                source_bindings=refs, union_run=str(union), mesh=mesh, local_run=str(local))


def _camera(record):
    import pycolmap
    return pycolmap.Camera(model=record['model'], width=record['width'], height=record['height'], params=record['params'])


def warp_evidence(context_view, review_root, refs):
    """Map only code1 external-gap hypotheses; no camera means no context_view."""
    root = Path(review_root).resolve(strict=True); v = context_view; i = v['decode_index']; stem = f'{i:06d}'
    result = _read(root/'RESULT.json', refs)
    _need(result.get('status') == 'AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED' and result.get('source_unchanged') is True and not result.get('mask_promotion'), 'review refused')
    _need(Path(result['source_run']).resolve() == Path(v['union_run']).resolve(), 'review union differs')
    decode = _read(root/'DECODE_BINDINGS.json', refs)
    matches = [x for x in decode['frames'] if x['decode_index'] == i]
    _need(len(matches) == 1, 'NO_BOUND_REVIEW_FRAME')
    n = matches[0]; ordinal = n['ordinal']
    _bind(decode['source_video']['path'], refs, decode['source_video'])
    native = _bind(root/'native'/f'{ordinal:06d}.png', refs, n['sha256'])
    canonical = _bind(root/'canonical'/f'{ordinal:06d}.jpg', refs, n['canonical_sha256'])
    e, f = v['frame'], v['mask_record']
    source = _bind(e['source_image_path'], refs, e['source_image_sha256'])
    pixels_equal = np.array_equal(_image(source), _image(canonical))
    if not pixels_equal:
        identity = v.get('frame_identity', {})
        _need(identity.get('status') == 'VERIFIED_BOUND_ORDINAL_PTS_RESAMPLING_CHAIN', 'SOURCE_RGB_PIXEL_MISMATCH / UNVERIFIED_FRAME_IDENTITY')
        _need(identity['source_video_sha256'] == decode['source_video']['sha256'] and
              identity['canonical_size'] == [_image(canonical).shape[1], _image(canonical).shape[0]], 'review/extraction video domain differs')
        for ref in identity['bindings']:
            _bind(ref['path'], refs, ref)
    gp = _read(root/'gaps/PLAN.json', refs); selected = [x for x in gp['frames'] if x['decode_index'] == i]
    _need(len(selected) == 1, 'gap frame identity mismatch'); selected = selected[0]
    _need(_bind(selected['union'], refs).read_bytes() == _bind(f['source_mask_path'], refs, f['source_mask_sha256']).read_bytes(), 'review source union differs')
    _bind(selected['image'], refs, n['canonical_sha256'])
    base = _binary(_bind(selected['base'], refs))
    evidence = _read(root/'evidence'/f'{stem}.json', refs)
    with np.load(_bind(root/'evidence'/f'{stem}.npz', refs), allow_pickle=False) as z:
        code, target = z['evidence_code'].copy(), z['target'].copy()
    with np.load(_bind(root/'gaps/arrays'/f'{stem}.npz', refs), allow_pickle=False) as z:
        external = z['narrow_external'].copy()
        # narrow_gap_review v1 explicitly evaluates the whole canonical raster
        # as its valid domain; unresolved background still cannot support votes.
        valid = ~z['unresolved_background'].copy()
    _need(code.shape == base.shape == target.shape == external.shape == valid.shape and target.dtype == external.dtype == valid.dtype == np.bool_ and np.isin(code, [0,1,2,3]).all(), 'evidence grid/type mismatch')
    _need(np.array_equal(target, external) and not (target & ((base != 0) | ~valid)).any() and
          np.isin(code[target], [1,2,3]).all(), 'external source background mismatch')
    rgb = _image(native)
    _need(hashlib.sha256(rgb.tobytes()).hexdigest() == evidence['input_sha256']['rgb_native'], 'native pixel identity mismatch')
    src, dst = _camera(f['source_camera']), _camera(f['target_camera'])
    union = _binary(f['source_mask_path'], base.shape)
    for key, array in (('base_canonical', base != 0), ('literal_union', union != 0)):
        _need(hashlib.sha256(array.tobytes()).hexdigest() == evidence['input_sha256'][key], 'evidence source pixels differ')
    _need(evidence['decode_index'] == i and evidence['stats']['source_target_pixels'] == int(target.sum()) and
          evidence['stats']['target_pixel_counts'] == {name: int((code[target] == k).sum()) for k, name in
          ((1,'SUPPORTED_RGB_CONSISTENCY'),(2,'WEAK_RGB_EVIDENCE'),(3,'UNKNOWN'))} and
          not code[~target].any(), 'evidence code/count identity mismatch')
    mask = warp_binary_mask(union, src, dst); support = warp_binary_mask(((code == 1) & external & valid).astype(np.uint8)*255, src, dst)
    valid_warp = warp_binary_mask(valid.astype(np.uint8)*255, src, dst)
    saved_valid = _binary(_bind(f['validity']['path'], refs, f['validity']))
    usable = (mask.valid != 0) & (saved_valid != 0)
    _need(np.array_equal(mask.mask, _binary(_bind(f['mask']['path'], refs, f['mask']))), 'remapped consumer mask differs')
    depth = None
    if v['depth_path'] is not None:
        depth = read_dmap(_bind(v['depth_path'], refs)); verify_depth(depth, v['camera_scene'])
    image_path = Path(v['surface_run'])/'bridge'/v['camera_scene']['name']
    _bind(image_path, refs)
    identity = dict(rgb_pixel_equal=pixels_equal, resampling_diff_known=not pixels_equal,
                    frame_identity=v.get('frame_identity', {}).get('status', 'EXACT_RGB_PIXELS'))
    return dict(mask=mask.mask != 0, validity=usable,
                supported_gap=(support.mask != 0) & usable & (support.valid != 0) & (valid_warp.mask != 0),
                depth=None if depth is None else depth.depth, image_path=str(image_path),
                identity=identity,
                rgb_pixel_equal=pixels_equal, resampling_diff_known=not pixels_equal,
                frame_identity=v.get('frame_identity', {}).get('status', 'EXACT_RGB_PIXELS'), geometry_accepted=False)
