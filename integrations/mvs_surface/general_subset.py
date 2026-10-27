"""Explicit, independently verified native view restriction; no SfM solving."""
import json
import math
from pathlib import Path
from .subset import need, read, _native, views, points, _tracks_valid
from .photo_inputs import _binding, _under, MODEL_BIN, _rows_by_index

SCHEMA = 'native-general-subset-v1'


def compare(source, child, retained):
    sv, dv = views(source), views(child)
    need(set(retained) <= set(sv) and len(retained) >= 2, 'at least two source views required')
    need(dv == {i: sv[i] for i in retained}, 'retained cameras/poses/frames differ')
    sp, dp = points(source), points(child)
    for pid,p in sp.items():
        need(all(math.isfinite(x) for x in p['xyz']), 'nonfinite source geometry')
        for iid,idx in p['track']:
            need(iid in sv and 0 <= idx < len(source.image(iid).points2D) and
                 int(source.image(iid).points2D[idx].point3D_id) == pid, 'source track reciprocal mismatch')
    for iid in sv:
        for idx,p in enumerate(source.image(iid).points2D):
            need(all(math.isfinite(x) for x in p.xy), 'nonfinite measured coordinates')
            if p.has_point3D():
                pid=int(p.point3D_id)
                need(pid in sp and (iid,idx) in sp[pid]['track'], 'source observation reciprocal mismatch')
    expected, removed, duplicates, observations = {}, [], [], []
    for pid, p in sorted(sp.items()):
        track = p['track']
        duplicate = len({i for i, _ in track}) != len(track)
        kept = [pair for pair in track if pair[0] in retained]
        if duplicate or len({i for i, _ in kept}) < 2:
            removed.append(pid)
            if duplicate: duplicates.append(pid)
            discarded = track
        else:
            expected[pid] = dict(p, track=kept)
            discarded = [pair for pair in track if pair[0] not in retained]
        observations.extend({'point3D_id': pid, 'image_id': i, 'point2D_idx': idx} for i, idx in discarded)
    need(dp == expected and bool(dp), 'point IDs/XYZ/RGB/tracks differ or empty points')
    for iid in retained:
        a, b = source.image(iid).points2D, child.image(iid).points2D
        need(len(a) == len(b), 'points2D length differs')
        for x, y in zip(a, b):
            need(x.xy.tolist() == y.xy.tolist(), 'measured coordinates changed')
            want = int(x.point3D_id) if x.has_point3D() and int(x.point3D_id) in expected else None
            got = int(y.point3D_id) if y.has_point3D() else None
            need(want == got, 'point association changed')
    need(child.is_valid(), 'derived consistency failed'); _tracks_valid(child, set(retained))
    return {'original_registered_ids': sorted(sv), 'derived_registered_ids': sorted(dv),
            'original_points': len(sp), 'derived_points': len(dp), 'removed_points': removed,
            'duplicate_image_points': duplicates, 'removed_observations': observations}


def evaluate(sfm_run, io_run, model_id, input_indices, derived_model, exclusion_reasons=None):
    sfm_run, io_run, derived_model = [Path(p).resolve(strict=True) for p in (sfm_run, io_run, derived_model)]
    sfm, sb = read(sfm_run/'run.json'); io, ib = read(io_run/'run.json')
    maps, mb = read(sfm_run/'frames_map.jsonl', True); frames, fb = read(io_run/'frames.jsonl', True)
    _rows_by_index(maps, 'SfM'); _rows_by_index(frames, 'IO')
    need(len(maps) == len(frames) and sfm.get('upstream_io_run_id') == io.get('run_id') and
         io.get('status') == 'SUCCESS' and sfm.get('status') in ('SUCCESS','PARTIAL'), 'run identity/status mismatch')
    need(io.get('contract_version') == 'photo-io-v1' and sfm.get('contract_version') in
         ('sfm-native-v1','sfm-native-profile-v1','sfm-native-object-v1','sfm-native-object-video-v1','sfm-native-object-pinhole-v1'), 'unsupported contracts')
    need(type(io.get('input_count')) is int and io['input_count'] == len(frames) and
         io.get('ok_count') == len(frames) and io.get('error_count') == 0 and
         sfm.get('counts', {}).get('input_rows') == len(maps), 'run counts mismatch')
    need(isinstance(model_id, str) and bool(model_id), 'explicit model_id required')
    selected = list(input_indices)
    need(selected and all(type(i) is int and 0 <= i < len(maps) for i in selected) and
         len(set(selected)) == len(selected), 'explicit unique input indices required')
    selected = sorted(selected)
    models = [m for m in sfm.get('models', []) if m.get('model_id') == model_id]
    need(len(models) == 1 and models[0].get('usable') is True, 'model missing/unusable')
    source_path = _under(sfm_run, models[0]['binary_path'])
    source, child = _native(source_path), _native(derived_model); sv = views(source)
    need(sorted(models[0].get('registered_image_ids', [])) == sorted(sv), 'declared native IDs differ')
    bindings = [sb, ib, mb, fb] + [_binding(p/f) for p in (source_path, derived_model) for f in MODEL_BIN]
    joined = []
    for m, f in zip(maps, frames):
        regs = [r for r in m.get('registrations', []) if r.get('model_id') == model_id]
        need(len(regs) <= 1, 'ambiguous registration')
        if not regs:
            need(m['input_index'] not in selected, 'selected input is unregistered'); continue
        r = regs[0]; iid = r.get('image_id')
        need(type(iid) is int and iid in sv and r.get('camera_id') == sv[iid]['camera_id'] and
             m.get('staged_image_name') == sv[iid]['name'], 'native mapping mismatch')
        need(f.get('status') == 'OK' and m.get('frame_id') == f.get('frame_id') and
             m.get('upstream_io_run_id') == io['run_id'] and m.get('upstream_image_output') == f.get('image_output'), 'IO mapping mismatch')
        canonical = _binding(_under(io_run, f['image_output'])); staged = _binding(_under(sfm_run, sv[iid]['name']))
        need(canonical['sha256'] == staged['sha256'] == m.get('canonical_image_sha256') and
             (f.get('canonical_width'), f.get('canonical_height')) == (sv[iid]['width'],sv[iid]['height']), 'canonical identity mismatch')
        bindings.extend((canonical,staged))
        joined.append({'input_index':m['input_index'], 'image_id':iid, 'frame_id':f['frame_id'], 'image_name':sv[iid]['name']})
    need({r['image_id'] for r in joined} == set(sv), 'unmapped native view')
    retained = sorted({r['image_id'] for r in joined if r['input_index'] in selected})
    # Aliases of a retained native view must all be explicitly selected.
    need(all((r['input_index'] in selected) == (r['image_id'] in retained) for r in joined), 'partially selected alias view')
    reasons = exclusion_reasons or {}
    excluded = [dict(r, reason=reasons.get(str(r['input_index']), reasons.get(r['input_index'], 'NOT_SELECTED'))) for r in joined if r['input_index'] not in selected]
    mapped_indices={r['input_index'] for r in joined}
    excluded.extend({'input_index':m['input_index'],'image_id':None,'frame_id':frames[m['input_index']].get('frame_id'),
                     'image_name':m.get('staged_image_name'),
                     'reason':reasons.get(str(m['input_index']),reasons.get(m['input_index'],'UNREGISTERED_IN_SELECTED_MODEL'))}
                    for m in maps if m['input_index'] not in mapped_indices)
    excluded.sort(key=lambda r:r['input_index'])
    need(all(isinstance(r['reason'],str) and r['reason'] for r in excluded), 'exclusion reason required')
    relation = compare(source,child,retained)
    bindings.extend(_binding(p) for p in (Path(__file__),Path(__file__).with_name('subset.py'),
                    Path(__file__).with_name('photo_inputs.py'),Path(__file__).parents[2]/'scripts/derive_general_subset.py'))
    for b in bindings: need(_binding(b['path']) == b, 'source/derived drift')
    return {'schema_version':SCHEMA, 'model_id':model_id, 'roots':{'sfm_run':str(sfm_run),'io_run':str(io_run),'derived_model':str(derived_model)},
            'original_model':str(source_path), 'input_indices':selected, 'selected_inputs':[r for r in joined if r['input_index'] in selected],
            'retained_native_ids':retained, 'removed_indices':[r['input_index'] for r in excluded], 'excluded_inputs':excluded,
            'exclusion_reasons':{str(r['input_index']):r['reason'] for r in excluded}, 'relation':relation, 'bindings':bindings,
            'scope':'Explicit mask-view restriction; incomplete target coverage and holes permitted; no complete-object claim'}


def derive(sfm_run, io_run, model_id, input_indices, output_dir, exclusion_reasons=None):
    input_indices=list(input_indices)
    output = Path(output_dir).resolve()
    source_run = Path(sfm_run).resolve(strict=True); io_root = Path(io_run).resolve(strict=True)
    need(not output.exists() and not output.is_relative_to(source_run) and not output.is_relative_to(io_root), 'new output outside source runs required')
    sfm, _ = read(source_run/'run.json'); models = [m for m in sfm.get('models',[]) if m.get('model_id') == model_id]
    need(len(models) == 1, 'explicit model missing')
    source = _under(source_run,models[0]['binary_path']); before = [_binding(source/f) for f in MODEL_BIN]
    maps, _ = read(source_run/'frames_map.jsonl',True)
    chosen = set(input_indices); ids = set()
    for m in maps:
        if m.get('input_index') in chosen:
            regs = [r for r in m.get('registrations',[]) if r.get('model_id') == model_id]
            need(len(regs) == 1 and type(regs[0].get('image_id')) is int, 'selected input registration missing')
            ids.add(regs[0]['image_id'])
    child = _native(source); need(len(ids) >= 2, 'at least two native views required')
    output.mkdir(parents=True); sparse = output/'sparse'; sparse.mkdir()
    try:
        for pid,p in list(child.points3D.items()):
            track = list(p.track.elements)
            if len({e.image_id for e in track}) != len(track): child.delete_point3D(pid)
        for iid,v in views(child).items():
            if iid not in ids:
                need(len(v['frame_data_ids']) == 1, 'excluded frame must contain one image')
                child.deregister_frame(v['frame_id'])
        child.write_binary(str(sparse))
        cert = evaluate(sfm_run,io_run,model_id,input_indices,sparse,exclusion_reasons)
        need(all(_binding(b['path']) == b for b in before), 'source drift')
        path = output/'subset_certificate.json'; path.write_text(json.dumps(cert,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return {'selected_sparse':str(sparse),'subset_certificate':{'path':str(path),'sha256':_binding(path)['sha256']},
                'selected_indices':cert['input_indices'],'missing_indices':cert['removed_indices'],'certificate':cert}
    except Exception as exc:
        (output/'FAILURE.json').write_text(json.dumps({'error':str(exc)},ensure_ascii=False),encoding='utf-8'); raise


def validate_reference(reference, *, derived_model=None, sfm_run=None, io_run=None, original_model=None):
    need(isinstance(reference,dict) and set(reference) == {'path','sha256'} and Path(reference['path']).is_absolute(), 'invalid certificate reference')
    cert, binding = read(reference['path']); need(binding['sha256'] == reference['sha256'] and cert.get('schema_version') == SCHEMA, 'certificate hash/schema differs')
    roots = cert.get('roots'); need(isinstance(roots,dict) and set(roots) == {'sfm_run','io_run','derived_model'}, 'invalid roots')
    need(all(isinstance(v,str) and Path(v).is_absolute() for v in roots.values()), 'absolute roots required')
    for k,v in dict(derived_model=derived_model,sfm_run=sfm_run,io_run=io_run).items():
        if v is not None: need(Path(v).resolve() == Path(roots[k]).resolve(), 'certificate belongs to another '+k)
    if original_model is not None: need(Path(original_model).resolve() == Path(cert['original_model']).resolve(), 'original model differs')
    actual = evaluate(**roots,model_id=cert['model_id'],input_indices=cert['input_indices'],exclusion_reasons=cert['exclusion_reasons'])
    need(actual == cert, 'certificate differs from independently recomputed relation')
    need(_binding(binding['path']) == binding, 'certificate drift')
    return dict(actual,certificate=binding,bindings=[binding,*actual['bindings']])
