"""Pure data fixtures: no native reader, model, or video execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from integrations.demo_pipeline import experimental_silhouette as e
from integrations.demo_pipeline import silhouette_review as s
from integrations.mvs_surface import experimental_inputs as native


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def ref(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': s._hash(path)}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    surface = root / 'outputs/surface'
    bridge = root / 'outputs/bridge'
    surface.mkdir(parents=True); bridge.mkdir()
    monkeypatch.setattr(s, 'ROOT', root)
    # Implementation bindings are part of the render contract.
    impl = root / 'integrations/demo_pipeline/experimental_silhouette.py'
    impl.parent.mkdir(parents=True); impl.write_text('fixture implementation')
    scene_path = bridge / 'scene.mvs'; scene_path.write_bytes(b'not native')
    images, plans, identities, outputs = [], [], [], []
    for index, name in [(17, 'images/z.png'), (0, 'images/a.png')]:
        photo = bridge / 'undistorted/images' / name
        photo.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (5, 5)).save(photo)
        mask, validity = [bridge / role / (Path(name).stem + suffix)
                          for role, suffix in [('masks', '.mask.png'), ('validity', '.validity.png')]]
        for path in (mask, validity):
            path.parent.mkdir(exist_ok=True)
            Image.new('L', (5, 5), 255).save(path)
        camera = {'global_id': index + 1, 'name': photo.relative_to(bridge).as_posix(),
                  'K': [[2, 0, 2], [0, 2, 2], [0, 0, 1]], 'R': np.eye(3).tolist(),
                  'C': [0, 0, 0], 'width': 5, 'height': 5}
        images.append(camera)
        plans.append({'image': copy.deepcopy(camera), 'image_file': ref(photo), 'mask': ref(mask),
                      'validity': ref(validity), 'foreground_pixels': 25, 'unknown_pixels': 0})
        identities.append({'input_index': index, 'native_name': name})
        outputs.append({'native_name': name, 'mask': ref(mask), 'validity': ref(validity)})
    spec_path = bridge / 'SURFACE_SPEC.json'
    sparse = bridge / 'sparse'; sparse.mkdir()
    producer = bridge / 'producer'; producer.mkdir()
    database = producer / 'database.db'; database.write_bytes(b'fixture')
    for name, document in [('PLAN.json', {'source_bindings': [], 'inputs': []}),
        ('CONFIG.json', {'models_root': str(producer), 'image_root': str(bridge / 'undistorted/images'),
                         'mask_root': str(bridge / 'masks'), 'image_names': [f['native_name'] for f in identities], 'arm': 'unmasked28'}),
        ('ENTRY.json', {'checkpoint': 'C:/sam3/sam3.pt'}), ('RESULT.json', {})]:
        write(producer / name, document)
    (producer / '0').mkdir()
    identity_path = producer / 'IDENTITY_SPEC.json'
    write(identity_path, {'schema_version': 'experimental-native-photo-source-v1', 'model_id': 0,
          'producer_plan': ref(producer / 'PLAN.json'), 'producer_config': ref(producer / 'CONFIG.json'),
          'producer_entry': ref(producer / 'ENTRY.json'), 'producer_result': ref(producer / 'RESULT.json'),
          'database': ref(database), 'model_files': [], 'mask_manifests': [], 'mask_selections': []})
    tool = bridge / 'tool.bin'; tool.write_bytes(b'fixture tool')
    write(bridge / 'PLAN.json', {'source_sparse': str(sparse), 'source_images': str(bridge / 'undistorted/images'),
                               'stages': [{}, {'argv': [str(tool)]}]})
    write(bridge / 'INPUT_BINDINGS.json', [])
    spec = {'schema_version': 'experimental-native-surface-input-v1', 'bridge_run': str(bridge), 'outputs': outputs,
            'identity_spec': ref(identity_path), 'bridge_bindings': []}
    write(spec_path, spec)
    plan = {'contract_version': 'mvs-surface-run-v1', 'scene': ref(scene_path), 'frames': plans,
            'arguments': {'input_origin': 'experimental-real', 'photo_source': str(spec_path),
                          'scene_path': str(scene_path), 'image_root': str(bridge),
                          'mask_root': str(bridge / 'masks'), 'validity_root': str(bridge / 'validity')}}
    write(surface / 'plan.json', plan)
    mesh = surface / 'work/__mvs_surface_raw.ply'
    mesh.parent.mkdir()
    mesh.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-1 -1 2\n1 -1 2\n0 1 2\n3 0 1 2\n')
    result = {'status': 'SURFACE_GENERATED_NOT_ACCEPTED', 'exit_code': 0, 'input_origin': 'caller_declared_experimental-real',
              'input_drift': [], 'stages': [{'name': 'mesh', 'status': 'GEOMETRY_READBACK_PASSED',
              'outputs': {'geometry': [dict(ref(mesh), role='pre_cleanup_surface', vertices=3, triangles=1)]}}]}
    write(surface / 'result.json', result)
    original = root / 'outputs/original/frames.jsonl'
    original.parent.mkdir(parents=True)
    original.write_text(''.join(json.dumps({'input_index': i}) + '\n' for i in [0, 8, 17, 900]), encoding='utf-8')
    validated = {'schema_version': 'experimental-native-surface-link-v1',
                 'status': 'EXPERIMENTAL_SOURCE_AND_WARP_BOUND', 'frames': identities,
                 'requested_indices': [0, 8, 17], 'bindings': [ref(scene_path), ref(original)]}
    monkeypatch.setattr(native, 'validate', lambda *args: copy.deepcopy(validated))
    monkeypatch.setattr(s, 'read_scene', lambda _: {'images': copy.deepcopy(images)})
    return surface, root / 'outputs/review', plan, spec, validated, images


def test_full_review_explicit_identity_and_exclusion(fixture):
    surface, output, *_ = fixture
    result = e.review_experimental(surface, output)
    assert result['completed_views'] == result['expected_views'] == 2
    assert result['selected_indices'] == [0, 17]
    assert result['excluded_indices'] == [8]
    assert result['source_kind'] == 'experimental-source'
    assert result['pipeline_acceptance'] is False
    scope = result['source_scope']['original_input_scope_records'][0]
    assert scope['explicit_input_indices'] == [0, 8, 17, 900]
    assert scope['outside_producer_indices'] == [900]
    assert not (surface / 'pipeline.json').exists()


@pytest.mark.parametrize('kind', ['duplicate_index', 'duplicate_name', 'missing_identity', 'missing_scene',
    'K_drift', 'R_drift', 'C_drift', 'global_id_drift', 'name_drift', 'mask_swap', 'invalid_status',
    'requested_scope', 'mesh_count', 'mesh_hash', 'mask_bytes', 'escape'])
def test_identity_counterexamples(fixture, kind):
    surface, output, plan, spec, validated, images = fixture
    if kind == 'duplicate_index': validated['frames'][1]['input_index'] = 17
    elif kind == 'duplicate_name': validated['frames'][1]['native_name'] = validated['frames'][0]['native_name']
    elif kind == 'missing_identity': validated['frames'].pop()
    elif kind == 'missing_scene': images.pop()
    elif kind.endswith('_drift') and kind.split('_')[0] in ('K', 'R', 'C'):
        key = kind.split('_')[0]
        if key == 'C': images[0][key][0] += 1
        else: images[0][key][0][0] += 1
    elif kind == 'global_id_drift': images[0]['global_id'] += 100
    elif kind == 'name_drift': images[0]['name'] = 'different.png'
    elif kind == 'mask_swap': spec['outputs'][0]['mask'] = spec['outputs'][1]['mask']; write(Path(plan['arguments']['photo_source']), spec)
    elif kind == 'invalid_status': validated['status'] = 'UNVERIFIED'
    elif kind == 'requested_scope': validated['requested_indices'] = [0]
    elif kind.startswith('mesh_'):
        path = surface / 'result.json'; result = s._read(path)
        result['stages'][0]['outputs']['geometry'][0]['vertices' if kind == 'mesh_count' else 'sha256'] = 4 if kind == 'mesh_count' else '0' * 64
        write(path, result)
    elif kind == 'mask_bytes': Path(spec['outputs'][0]['mask']['path']).write_bytes(b'drift')
    elif kind == 'escape': plan['arguments']['photo_source'] = str(surface.parents[2] / 'external.json'); write(Path(plan['arguments']['photo_source']), spec); write(surface / 'plan.json', plan)
    with pytest.raises((ValueError, KeyError)):
        e.review_experimental(surface, output)
    assert not output.exists()


def test_drift_rejected_during_render(fixture, monkeypatch):
    surface, output, plan, *_ = fixture
    original = s.raster
    def raster(*args, **kwargs):
        Path(plan['frames'][0]['image_file']['path']).write_bytes(b'drift')
        return original(*args, **kwargs)
    monkeypatch.setattr(s, 'raster', raster)
    with pytest.raises(Exception): e.review_experimental(surface, output)
    assert s._read(output / 'RESULT.json')['status'] == 'INCOMPLETE'


def test_limit_preflight_before_native(fixture, monkeypatch):
    surface, _, plan, *_ = fixture
    huge = surface / 'oversized.bin'
    with huge.open('wb') as stream: stream.seek(64 << 20); stream.write(b'x')
    spec = s._read(plan['arguments']['photo_source'])
    spec['bridge_bindings'] = [ref(huge)]; write(Path(plan['arguments']['photo_source']), spec)
    monkeypatch.setattr(native, 'validate', lambda *args: pytest.fail('native reader invoked'))
    with pytest.raises(ValueError, match='64 MiB'): e.inspect(surface)


def test_unrelated_checkpoint_and_environment_strings_are_not_followed(fixture, monkeypatch):
    surface, _, plan, *_ = fixture
    ignored = surface.parents[2] / 'sam3.pt'
    ignored.write_bytes(b'outside project: must not read or hash')
    original_add = s.Bindings.add
    def add(self, path, *args):
        assert Path(path).resolve() != ignored.resolve() and 'sam3.pt' not in str(path) and 'unrelated-tool' not in str(path)
        return original_add(self, path, *args)
    monkeypatch.setattr(s.Bindings, 'add', add)
    plan['checkpoint'] = str(ignored)
    plan['environment'] = {'PATH': 'C:/unrelated-tool.exe'}
    write(surface / 'plan.json', plan)
    e.inspect(surface)


def test_consumed_native_directory_cannot_escape_before_validation(fixture, monkeypatch):
    surface, _, plan, *_ = fixture
    spec_path = Path(plan['arguments']['photo_source']); spec = s._read(spec_path)
    identity_path = Path(spec['identity_spec']['path']); identity = s._read(identity_path)
    config_path = Path(identity['producer_config']['path']); config = s._read(config_path)
    outside = surface.parents[2] / 'outside_model'; outside.mkdir()
    config['models_root'] = str(outside); write(config_path, config)
    identity['producer_config'] = ref(config_path); write(identity_path, identity)
    spec['identity_spec'] = ref(identity_path); write(spec_path, spec)
    monkeypatch.setattr(native, 'validate', lambda *args: pytest.fail('native reader invoked'))
    with pytest.raises(ValueError, match='ROOT'): e.inspect(surface)


def test_real_result_origin_is_distinct_from_plan_argument(fixture):
    surface, output, *_ = fixture
    path = surface / 'result.json'; result = s._read(path)
    result['input_origin'] = 'experimental-real'; write(path, result)
    with pytest.raises(ValueError, match='explicitly experimental'): e.review_experimental(surface, output)


def test_count_only_record_does_not_invent_indices(fixture):
    surface, _, _, _, validated, _ = fixture
    old = surface.parent / 'count_only/run.json'
    write(old, {'contract_version': 'sfm-native-object-v1', 'counts': {'input_rows': 36}})
    validated['bindings'] = [ref(old)]
    state, *_ = e.inspect(surface)
    assert state['source_scope']['original_input_scope_records'] == []


@pytest.mark.parametrize('arguments', [[], ['--run', 'x', '--experimental-surface', 'y']])
def test_cli_source_group_required_and_mutually_exclusive(arguments, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/review_surface_silhouettes.py'
    module_spec = importlib.util.spec_from_file_location('review_cli', script)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [str(script), '--output', 'out', *arguments])
    with pytest.raises(SystemExit) as exc: module.main()
    assert exc.value.code == 2
