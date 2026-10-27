"""Read an explicitly experimental surface; never synthesize a pipeline receipt."""
from pathlib import Path

from PIL import Image

from . import silhouette_review as review


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bind_record(bindings, record):
    _need(isinstance(record, dict) and {'path', 'bytes', 'sha256'} <= set(record),
          'complete source byte record required')
    path = bindings.add(record['path'], record['sha256'])
    _need(type(record['bytes']) is int and path.stat().st_size == record['bytes'],
          'declared source size differs')
    return path


def _preflight(bindings, spec_path):
    """Whitelist precisely the source roles reopened by experimental_inputs.validate.

    Bound JSON records are opaque unless their fields are explicitly consumed below.
    Environment, checkpoint, historical argv and unrelated tool references are ignored.
    """
    from integrations.mvs_masks import manifest_io as mio
    def document(record):
        return review._read(_bind_record(bindings, record))
    def directory(path):
        path = Path(path).resolve(strict=True)
        _need(path.is_dir() and path.is_relative_to(bindings.root), 'source directory escapes ROOT')
        return path
    def files(path, recursive=False):
        root = directory(path)
        for path in (root.rglob('*') if recursive else root.iterdir()):
            if path.is_file(): bindings.add(path)
    visited = set()
    def identity(record):
        path = _bind_record(bindings, record)
        _need(path not in visited, 'identity cycle')
        visited.add(path)
        spec = review._read(path)
        if spec['schema_version'] == 'experimental-native-derived-photo-source-v1':
            identity(spec['parent_identity_spec'])
            document(spec['certificate'])
            for row in spec['code_bindings'] + spec['model_files']: _bind_record(bindings, row)
            files(spec['derived_sparse'])
            return
        _need(spec['schema_version'] == 'experimental-native-photo-source-v1', 'source identity schema differs')
        plan = document(spec['producer_plan']); config = document(spec['producer_config'])
        document(spec['producer_entry']); document(spec['producer_result'])
        _bind_record(bindings, spec['database'])
        archived = {row['original_path']: row['archive'] for row in spec.get('historical_code_copies', [])}
        for row in spec.get('historical_code_copies', []): _bind_record(bindings, row['archive'])
        for row in plan['source_bindings']:
            if row['path'] not in archived: _bind_record(bindings, row)
        for row in spec['model_files']: _bind_record(bindings, row)
        files(directory(config['models_root']) / str(spec['model_id']))
        rgb_root = directory(config['image_root']); directory(config['mask_root'])
        for name in config['image_names']: bindings.add(rgb_root / name)
        for row in plan['inputs']:
            if row['arm'] == config['arm']:
                _bind_record(bindings, row['source']); _bind_record(bindings, row['copy'])
        for row in spec['mask_manifests']:
            path = _bind_record(bindings, row)
            manifest = mio.parse_object_manifest(path)
            io = mio.resolve_source_io(manifest['source_io'], manifest['manifest_path'])
            for key in ('run_json', 'frames_jsonl'): _bind_record(bindings, io[key])
            directory(io['path'])
            frames = review.read_jsonl_strict(io['frames_jsonl']['path'])
            selected = {s['input_index'] for s in spec['mask_selections']}
            for frame in frames:
                if frame['input_index'] in selected: bindings.add(Path(io['path']) / frame['image_output'])
    spec_path = bindings.add(spec_path)
    spec = review._read(spec_path)
    identity(spec['identity_spec'])
    bridge = directory(spec['bridge_run'])
    for row in spec['bridge_bindings']: _bind_record(bindings, row)
    # validate checks complete bridge file coverage and all three sparse directories.
    files(bridge, recursive=True)
    bridge_plan = review._read(bindings.add(bridge / 'PLAN.json'))
    files(bridge_plan['source_sparse']); directory(bridge_plan['source_images'])
    inputs_path = bindings.add(bridge / 'INPUT_BINDINGS.json')
    for row in mio.load_json_array_strict(inputs_path, 'bridge inputs'): _bind_record(bindings, row)
    bindings.add(bridge_plan['stages'][1]['argv'][0])
    for row in spec['outputs']:
        for role in ('mask', 'validity'): _bind_record(bindings, row[role])
    _need(len(bindings.records) <= 4096, 'source closure exceeds 4096 files')


def join_views(plan, spec, validated, scene, bindings):
    """Join using authenticated native names, never positional or sorted filenames."""
    native_root = Path(spec['bridge_run']).resolve() / 'undistorted' / 'images'
    def unique(rows, key, label):
        _need(isinstance(rows, list) and 1 <= len(rows) <= 96, label + ' coverage invalid')
        result = {}
        for row in rows:
            value = row[key]
            _need(value not in result, label + ' duplicate identity')
            result[value] = row
        return result
    identities = unique(validated['frames'], 'native_name', 'validated frame')
    outputs = unique(spec['outputs'], 'native_name', 'warped output')
    cameras = unique(scene['images'], 'global_id', 'scene')
    planned = unique([p['image'] for p in plan['frames']], 'global_id', 'planned scene')
    _need(set(identities) == set(outputs) and set(cameras) == set(planned)
          and len(identities) == len(cameras) == len(plan['frames']), 'all-view bijection differs')
    indices, names, prepared = set(), set(), []
    scene_path = Path(plan['scene']['path']).resolve()
    for p in plan['frames']:
        camera = cameras[p['image']['global_id']]
        for key in ('global_id', 'name', 'K', 'R', 'C', 'width', 'height'):
            _need(camera[key] == p['image'][key], 'actual MVSI camera/name drift')
        image = _bind_record(bindings, p['image_file'])
        _need(image == (scene_path.parent / camera['name']).resolve(), 'MVSI image path differs')
        matches = [name for name in identities if (native_root / name).resolve() == image]
        _need(len(matches) == 1, 'native image/MVSI name is not bijective')
        name = matches[0]
        identity = identities[name]
        index = identity['input_index']
        _need(type(index) is int and index >= 0 and index not in indices and name not in names,
              'duplicate/invalid input index or native name')
        indices.add(index); names.add(name)
        size = [camera['width'], camera['height']]
        arrays = []
        for role in ('mask', 'validity'):
            _need(p[role] == outputs[name][role], 'plan and SPEC label identity differs')
            path = _bind_record(bindings, outputs[name][role])
            expected_root = Path(plan['arguments'][role + '_root']).resolve()
            suffix = '.mask.png' if role == 'mask' else '.validity.png'
            _need(path == (expected_root / (Path(name).stem + suffix)).resolve(), 'label path differs')
            arrays.append(review._binary(path, size))
        sam, valid = arrays
        _need(not (sam & ~valid).any() and int(sam.sum()) == p['foreground_pixels'],
              'foreground count/validity conflict')
        _need(int((~valid).sum()) == p['unknown_pixels'], 'unknown count conflict')
        with Image.open(image) as photo:
            _need(photo.size == tuple(size), 'photo dimensions differ')
        prepared.append(({'input_index': index, 'image_name': name}, camera, sam, valid, image))
    _need(names == set(identities), 'unreviewed selected native view')
    return sorted(prepared, key=lambda row: row[0]['input_index'])


def inspect(surface):
    surface = Path(surface).resolve(strict=True)
    _need(surface.is_relative_to(review.ROOT), 'experimental surface escapes project ROOT')
    bindings = review.Bindings(review.ROOT)
    plan_path, result_path = [bindings.add(surface / name) for name in ('plan.json', 'result.json')]
    plan, result = review._read(plan_path), review._read(result_path)
    _need(plan.get('contract_version') == 'mvs-surface-run-v1'
          and plan['arguments'].get('input_origin') == 'experimental-real'
          and result.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED'
          and type(result.get('exit_code')) is int and result['exit_code'] == 0
          and result.get('input_origin') == 'caller_declared_experimental-real' and result.get('input_drift') == [],
          'successful explicitly experimental surface required')
    args = plan['arguments']
    spec_path = bindings.add(args['photo_source'])
    spec = review._read(spec_path)
    _need(spec.get('schema_version') == 'experimental-native-surface-input-v1', 'experimental SPEC required')
    scene_path = _bind_record(bindings, plan['scene'])
    _need(scene_path == Path(args['scene_path']).resolve(), 'plan scene argument differs')
    bridge = Path(spec['bridge_run']).resolve(strict=True)
    _need(bridge.is_relative_to(review.ROOT), 'bridge escapes project ROOT')
    _preflight(bindings, spec_path)
    for frame in plan['frames']:
        for role in ('image_file', 'mask', 'validity'): _bind_record(bindings, frame[role])
    from integrations.mvs_surface import experimental_inputs
    validated = experimental_inputs.validate(spec_path, scene_path, args['image_root'],
                                               args['mask_root'], args['validity_root'])
    _need(validated.get('schema_version') == 'experimental-native-surface-link-v1'
          and validated.get('status') == 'EXPERIMENTAL_SOURCE_AND_WARP_BOUND',
          'identity/warp validation did not complete')
    for record in validated['bindings']:
        _bind_record(bindings, record)
    scene = review.read_scene(scene_path)
    prepared = join_views(plan, spec, validated, scene, bindings)
    mesh_path = (surface / 'work/__mvs_surface_raw.ply').resolve()
    declarations = [g for stage in result['stages'] if stage.get('name') == 'mesh'
                    and stage.get('status') == 'GEOMETRY_READBACK_PASSED'
                    for g in stage.get('outputs', {}).get('geometry', [])
                    if g.get('role') == 'pre_cleanup_surface' and Path(g['path']).resolve() == mesh_path]
    _need(len(declarations) == 1, 'unique mesh-stage raw geometry declaration required')
    declaration = declarations[0]
    _bind_record(bindings, declaration)
    mesh = review.read_ply(mesh_path, max_bytes=64 << 20, max_vertices=500000, max_faces=1000000)
    _need(mesh['faces'] and len(mesh['vertices']) == declaration['vertices']
          and len(mesh['faces']) == declaration['triangles'], 'mesh readback counts differ')
    selected = {p[0]['input_index'] for p in prepared}
    requested = validated['requested_indices']
    _need(isinstance(requested, list) and len(set(requested)) == len(requested)
          and selected <= set(requested), 'requested experimental scope differs')
    scope = {'producer_requested_indices': requested, 'reviewed_indices': sorted(selected),
             'producer_unregistered_indices': sorted(set(requested) - selected),
             'outside_producer_scope': 'Original video frames outside the producer selected subset are not evaluated.',
             'quality_evidence': 'SAM_INTERNAL_CONSISTENCY_ONLY', 'pipeline_acceptance': False}
    original_scopes = []
    for record in validated['bindings']:
        path = Path(record['path'])
        if path.name == 'frames.jsonl':
            rows = review.read_jsonl_strict(path)
            indices = [row['input_index'] for row in rows]
            _need(all(type(i) is int and i >= 0 for i in indices) and len(set(indices)) == len(indices),
                  'bound photoIO indices invalid')
            _need(set(requested) <= set(indices), 'producer indices outside bound photoIO frames')
            original_scopes.append({'source_record': record, 'declared_input_rows': len(rows),
                'explicit_input_indices': sorted(indices),
                'outside_producer_indices': sorted(set(indices) - set(requested)),
                'all_unreviewed_indices': sorted(set(indices) - selected),
                'evidence': 'Explicit input_index from producer-bound photoIO frames; excluded frames not evaluated.'})
    scope['original_input_scope_records'] = original_scopes
    _need(bindings.unchanged(), 'source drift after inspection')
    # Rendering metadata only; this is never serialized as a pipeline receipt.
    metadata = {'stages': {'surface': {'missing_indices': scope['producer_unregistered_indices']}},
                'source_scope': scope}
    return metadata, mesh, prepared, bindings


def review_experimental(surface, output, *, seconds=180):
    return review.render_review(surface, output, inspect, seconds=seconds,
                                source_kind='experimental-source')
