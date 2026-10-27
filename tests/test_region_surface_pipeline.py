"""TEST_DATA: controller gates only; all native readers and launches are mocked."""
import sys
import json
from types import SimpleNamespace

import pytest

from integrations.video_masks import region_surface as surface


def _model(point_count=100, invalid_count=0):
    points = {}
    for pid in range(point_count):
        views = [1, 1, 2] if pid < invalid_count else [1, 2]
        points[pid] = SimpleNamespace(track=SimpleNamespace(
            elements=[SimpleNamespace(image_id=i) for i in views]))
    return SimpleNamespace(points3D=points, num_reg_images=lambda: 2,
                           num_points3D=lambda: point_count,
                           compute_num_observations=lambda: sum(len(p.track.elements) for p in points.values()))


def test_component_choice_is_deterministic_and_never_merges():
    models = [dict(model_id=2, registered_views=8, points=80),
              dict(model_id=1, registered_views=8, points=80),
              dict(model_id=0, registered_views=7, points=900)]
    assert surface.choose_model(models) is models[1]
    assert surface.choose_model(models, 0) is models[2]
    with pytest.raises(ValueError, match='no reconstructed component'):
        surface.choose_model([])
    with pytest.raises(ValueError, match='uniquely present'):
        surface.choose_model(models, 9)
    with pytest.raises(ValueError, match='uniquely present'):
        surface.choose_model(models + [models[0]], 2)


@pytest.mark.parametrize('point_count,invalid_count,allowed', [
    (99, 1, False), (100, 1, True), (199, 2, False),
    (200, 2, True), (2059, 20, True), (2059, 21, False), (99, 0, True),
])
def test_structural_exclusion_is_strictly_at_most_one_percent(point_count, invalid_count, allowed):
    model = _model(point_count, invalid_count)
    if allowed:
        assert surface.exclusion_candidates(model) == list(range(invalid_count))
    else:
        with pytest.raises(ValueError, match='1 percent'):
            surface.exclusion_candidates(model)
    assert len(model.points3D) == point_count


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    source = root/'outputs/source'
    binary = source/'geometry/models/0/binary'
    binary.mkdir(parents=True)
    (binary/'points3D.bin').write_bytes(b'TEST_DATA, never opened by a native reader')
    (source/'input').mkdir()
    surface.save(source/'geometry/RESULT.json', dict(models=[dict(
        model_id=0, registered_views=2, points=100, binary_path=str(binary))]))
    surface.save(source/'geometry/PLAN.json', dict(frames=[]))
    surface.save(source/'input/PLAN.json', dict(timeline=[]))
    python = root/'runtime/python.exe'
    python.parent.mkdir()
    python.write_bytes(b'TEST_DATA, never executed')
    tools = root/'tools'
    tools.mkdir()
    (tools/'InterfaceCOLMAP.exe').write_bytes(b'TEST_DATA, never executed')
    monkeypatch.setattr(surface, 'ROOT', root)
    monkeypatch.setattr(surface, 'LIGHT_REL', 'runtime/python.exe')
    monkeypatch.setattr(surface, 'MVS_REL', 'tools')
    monkeypatch.setattr(surface, 'code_bindings', lambda: [])
    monkeypatch.setitem(sys.modules, 'pycolmap', SimpleNamespace(Reconstruction=lambda path: _model()))
    from integrations.mvs_surface import runner
    plan_calls = []
    def plan(*args, **kwargs):
        plan_calls.append((args, kwargs))
        return dict(TEST_DATA=True, arguments=kwargs)
    monkeypatch.setattr(runner, 'make_plan', plan)
    calls = []
    options = dict(fail='export', kind='process', drift=False, gate_accepted=False, gate_missing=None,
                   local_accepted=False,local_missing=None)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        surface.save(path, value)
    def invoke(repo, out, name, argv, *, seconds, memory):
        calls.append(dict(name=name, argv=argv, seconds=seconds, memory=memory))
        process_failed = options['fail'] == name and options['kind'] == 'process'
        business_failed = options['fail'] == name and options['kind'] == 'business'
        if not process_failed:
            if name == 'export':
                write(out/'export/EXPORT.json', dict(status='FAILED' if business_failed else 'EXPORTED_NOT_GEOMETRY_ACCEPTED',
                      source_bindings=[surface.binding(source/'input/PLAN.json')]))
                write(out/'export/DERIVATION.json', dict(TEST_DATA=True,
                      after=dict(images=2, points=100, observations=200)))
            elif name == 'camera_domain':
                write(out/'CAMERA_DOMAIN.json', dict(status='CAMERA_DOMAIN_REJECTED' if business_failed else 'CAMERA_DOMAIN_VALID', source_unchanged=True))
            elif name == 'bridge':
                write(out/'bridge/RESULT.json', dict(status='FAIL' if business_failed else 'PASS', input_bindings_unchanged=True))
            elif name == 'masks':
                write(out/'masks/RESULT.json', dict(coverage_status='COMPLETE', adapter_source_chain='VERIFIED',
                      available_views=1 if business_failed else 2, surface_image_root=str(out/'bridge'),
                      source_bindings=[surface.binding(out/'export/EXPORT.json')]))
            elif name == 'surface':
                write(out/'surface/result.json', dict(status='FAILED' if business_failed else 'SURFACE_GENERATED_NOT_ACCEPTED'))
                if not business_failed:
                    (out/'surface/work').mkdir(parents=True)
                    for filename in ('__mvs_dense.ply','__mvs_surface_raw.ply'):
                        (out/'surface/work'/filename).write_bytes(b'TEST_DATA never parsed')
            elif name == 'blender':
                write(out/'blender/RESULT.json',dict(status='SUCCESS'))
            elif name == 'point_density':
                write(out/'point_density/RESULT.json',dict(source_unchanged=True,
                      status='BELOW_DENSITY_TARGET',dense_visibility_coverage='NOT_EVALUATED'))
                (out/'point_density/index.html').write_text('TEST_DATA')
            elif name == 'surface_support':
                write(out/'support_audit/RESULT.json',dict(source_unchanged=not business_failed,
                      status='DIAGNOSTIC_ONLY',geometry_accepted=False))
                (out/'support_audit/index.html').write_text('TEST_DATA')
            elif name == 'face_mask_gate':
                write(out/'face_mask_gate/RESULT.json',dict(source_unchanged=not business_failed,
                      status='CANDIDATE_SURFACE_NOT_ACCEPTED',geometry_accepted=options['gate_accepted']))
                for filename in ('index.html','comparison_viewer.html','retained_viewer.html','retained.ply','quarantined.ply','FACE_LEDGER.csv'):
                    if filename != options['gate_missing']:
                        (out/'face_mask_gate'/filename).write_text('TEST_DATA')
            elif name == 'face_local':
                write(out/'face_local/RESULT.json',dict(source_unchanged=not business_failed,
                      status='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED',geometry_accepted=options['local_accepted']))
                for filename in ('index.html','comparison_viewer.html','retained_viewer.html','retained.ply',
                                 'quarantined.ply','FACE_LEDGER.csv','LINEAGE.npz'):
                    if filename != options['local_missing']:
                        (out/'face_local'/filename).write_text('TEST_DATA')
            elif name == 'boundary_geometry':
                record=dict(status='BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED',
                            source_unchanged=not business_failed,geometry_accepted=False,geometry_changed=False)
                record.update(options.get('boundary_record',{}))
                write(out/'boundary_geometry/RESULT.json',record)
                if not options.get('boundary_missing_report'):
                    (out/'boundary_geometry/index.html').write_text('TEST_DATA')
            elif name == 'shape_support':
                candidate=argv[argv.index('--candidate-run')+1] if '--candidate-run' in argv else None
                record=dict(schema='shape-support-audit-v1',status='NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE',
                            source_unchanged=not business_failed,geometry_changed=False,
                            source_surface_run=str(out),candidate_run=candidate,candidate_promotable=candidate is not None)
                record.update(options.get('shape_record',{}))
                write(out/'shape_support/RESULT.json',record)
                if not options.get('shape_missing_report'):
                    (out/'shape_support/index.html').write_text('TEST_DATA')
            else:
                raise AssertionError('No later native stage may be reached: '+name)
        if options['drift']:
            (binary/'points3D.bin').write_bytes(b'TEST_DATA mutated source')
        return dict(status='completed', root_exit_code=1 if process_failed else 0,
                    reclaim_confirmed=True, errors=[], log_errors=[])
    monkeypatch.setattr(surface, '_invoke', invoke)
    return SimpleNamespace(root=root, source=source, out=root/'outputs/result',
                           calls=calls, options=options, plan_calls=plan_calls)


@pytest.mark.parametrize('failed_stage', ['export', 'camera_domain', 'bridge', 'masks', 'surface'])
@pytest.mark.parametrize('kind', ['process', 'business'])
def test_failure_stops_before_any_later_launch(controller, failed_stage, kind):
    controller.options.update(fail=failed_stage, kind=kind)
    result = surface.run(controller.source, controller.out)
    order = ['export', 'camera_domain', 'bridge', 'masks', 'surface']
    assert [x['name'] for x in controller.calls] == order[:order.index(failed_stage)+1]
    assert result['status'] == 'PARTIAL_OR_FAILED'
    assert result['stages'][-1]['status'] == ('FAILED' if kind == 'process' else 'BUSINESS_GATE_FAILED')
    assert all(x['status'] == 'BUSINESS_GATE_PASSED' for x in result['stages'][:-1])
    assert result['geometry_accepted'] is False
    assert not result['binding_drift']
    assert (controller.out/'RESULT.json').is_file()
    if failed_stage == 'surface':
        _, frozen = controller.plan_calls[0]
        assert frozen == dict(total_seconds=1200, stage_seconds=900, memory_bytes=8 << 30,
                              output_bytes=2 << 30, input_origin='unverified', ignore_label=0, photo_source=None)


def test_failed_output_cannot_be_retried_or_overwritten(controller):
    surface.run(controller.source, controller.out)
    record = (controller.out/'RESULT.json').read_bytes()
    with pytest.raises(ValueError, match='fresh disjoint'):
        surface.run(controller.source, controller.out)
    assert len(controller.calls) == 1
    assert (controller.out/'RESULT.json').read_bytes() == record


def test_surface_automatically_audits_density_without_claiming_quality(controller,monkeypatch):
    controller.options.update(fail=None)
    from integrations.viewer import export
    def fake_viewer(spec,destination,profile):
        destination.write_text('TEST_DATA viewer')
        return dict(TEST_DATA=True)
    monkeypatch.setattr(export,'export_html',fake_viewer)
    result=surface.run(controller.source,controller.out)
    assert [r['name'] for r in controller.calls][-7:-2]==['blender','point_density','surface_support','face_mask_gate','face_local']
    assert [r['name'] for r in controller.calls][-2:]==['boundary_geometry','shape_support']
    assert result['status']=='SURFACE_GENERATED_NOT_ACCEPTED'
    assert result['point_density_status']=='BELOW_DENSITY_TARGET'
    assert result['point_density_target_met'] is False
    assert result['dense_visibility_coverage']=='NOT_EVALUATED'
    assert result['geometry_accepted'] is False
    assert result['surface_support_status']=='DIAGNOSTIC_ONLY'
    assert result['stages'][-3]['status']=='LOCAL_CANDIDATE_DERIVED_NOT_QUALITY_ACCEPTED'
    assert result['stages'][-2]['status']=='BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED'
    assert result['stages'][-1]['status']=='NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE'
    assert result['candidate_promoted'] is True
    assert result['face_mask_gate_status']=='CANDIDATE_SURFACE_NOT_ACCEPTED'
    assert result['preferred_viewer']==str(controller.out/'face_local/retained_viewer.html')
    assert result['artifacts']['单独修正版候选（未验收）']==result['preferred_viewer']
    assert result['artifacts']['原始面与候选隔离面对照（未验收）']==str(controller.out/'face_mask_gate/comparison_viewer.html')
    assert result['face_local_status']=='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED'
    assert result['artifacts']['上一阶段边缘候选（未验收）']==str(controller.out/'face_mask_gate/retained_viewer.html')
    assert controller.calls[-3]['seconds']==480
    assert controller.calls[-3]['memory']==4<<30
    assert controller.calls[-4]['seconds']==240
    assert controller.calls[-4]['memory']==3<<30
    assert controller.calls[-2]['seconds']==600
    assert controller.calls[-2]['memory']==4<<30
    assert controller.calls[-1]['seconds']==600 and controller.calls[-1]['memory']==4<<30
    argv=controller.calls[-3]['argv']
    assert argv[argv.index('--gate')+1]==str(controller.out/'face_mask_gate')
    assert '默认查看' in (controller.out/'index.html').read_text(encoding='utf8')
    bindings=json.loads((controller.out/'STAGE_BINDINGS.json').read_text(encoding='utf8'))
    assert any(r['path']==str(controller.out/'face_mask_gate/FACE_LEDGER.csv') for r in bindings)
    assert any(r['path']==str(controller.out/'face_local/LINEAGE.npz') for r in bindings)


def test_support_worker_routes_without_native_model_reconstruction(tmp_path,monkeypatch):
    from integrations.video_masks import surface_support
    calls=[]
    monkeypatch.setattr(surface,'run',lambda *a: pytest.fail('surface reconstruction must not run'))
    monkeypatch.setattr(surface_support,'run',lambda *a: calls.append(a) or {'status':'DIAGNOSTIC_ONLY'})
    assert surface.main(['support-worker','--source',str(tmp_path/'source'),
                         '--output',str(tmp_path/'audit')])==0
    assert calls==[(tmp_path/'source',tmp_path/'audit')]


def test_face_mask_worker_routes_without_native_model_reconstruction(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(surface,'run',lambda *a: pytest.fail('surface reconstruction must not run'))
    monkeypatch.setitem(sys.modules,'integrations.video_masks.face_mask_adapter',SimpleNamespace(
        run=lambda *a: calls.append(a) or dict(status='CANDIDATE_SURFACE_NOT_ACCEPTED')))
    assert surface.main(['face-mask-worker','--source',str(tmp_path/'source'),
                         '--output',str(tmp_path/'gate')])==0
    assert calls==[(tmp_path/'source',tmp_path/'gate')]


def test_face_local_worker_routes_without_native_model_reconstruction(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(surface,'run',lambda *a: pytest.fail('surface reconstruction must not run'))
    monkeypatch.setitem(sys.modules,'integrations.video_masks.local_surface_adapter',SimpleNamespace(
        run=lambda *a: calls.append(a) or dict(status='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED')))
    assert surface.main(['face-local-worker','--source',str(tmp_path/'source'),'--gate',str(tmp_path/'gate'),
                         '--output',str(tmp_path/'local')])==0
    assert calls==[(tmp_path/'source',tmp_path/'gate',tmp_path/'local')]


def test_face_local_worker_requires_explicit_gate_before_any_work(tmp_path,monkeypatch):
    with pytest.raises(SystemExit):
        surface.main(['face-local-worker','--source',str(tmp_path/'source'),'--output',str(tmp_path/'local')])
    assert not (tmp_path/'local').exists()


@pytest.mark.parametrize('kind',['process','business'])
def test_local_failure_retains_raw_viewer_without_claiming_completion(controller,monkeypatch,kind):
    from integrations.viewer import export
    monkeypatch.setattr(export,'export_html',lambda spec,destination,profile: {})
    controller.options.update(fail='face_local',kind=kind)
    result=surface.run(controller.source,controller.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['geometry_accepted'] is False
    assert result['face_local_status']=='NOT_EVALUATED'
    assert result['face_mask_gate_status']=='CANDIDATE_SURFACE_NOT_ACCEPTED'
    assert result['preferred_viewer']==str(controller.out/'mesh_viewer.html')
    assert result['artifacts']['上一阶段边缘候选（未验收）']==str(controller.out/'face_mask_gate/retained_viewer.html')
    assert result['candidate_promoted'] is False
    assert result['stages'][-2]['name']=='face_local'
    assert result['stages'][-2]['status']==('FAILED' if kind=='process' else 'BUSINESS_GATE_FAILED')
    assert result['stages'][-1]['name']=='shape_support'
    assert '局部边界处理记录' not in result['artifacts']
    assert result['error']


@pytest.mark.parametrize('options',[dict(local_accepted=True),dict(local_missing='LINEAGE.npz'),
                                 dict(local_missing='retained_viewer.html')])
def test_local_candidate_requires_unaccepted_contract_and_real_lineage(controller,monkeypatch,options):
    from integrations.viewer import export
    monkeypatch.setattr(export,'export_html',lambda spec,destination,profile: {})
    controller.options.update(fail=None,**options)
    result=surface.run(controller.source,controller.out)
    assert result['face_local_status']=='NOT_EVALUATED'
    assert result['preferred_viewer']==str(controller.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False
    assert result['stages'][-2]['status']=='BUSINESS_GATE_FAILED'
    assert result['stages'][-1]['name']=='shape_support'


@pytest.mark.parametrize('kind',['process','business'])
def test_face_mask_failure_does_not_publish_filtered_or_accepted_model(controller,monkeypatch,kind):
    from integrations.viewer import export
    monkeypatch.setattr(export,'export_html',lambda spec,destination,profile: {})
    controller.options.update(fail='face_mask_gate',kind=kind)
    result=surface.run(controller.source,controller.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['face_mask_gate_status']=='NOT_EVALUATED'
    assert result['geometry_accepted'] is False
    assert result['preferred_viewer']==str(controller.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False
    assert result['stages'][-2]['status']==('FAILED' if kind=='process' else 'BUSINESS_GATE_FAILED')
    assert result['stages'][-1]['name']=='shape_support'
    assert result['error']


@pytest.mark.parametrize('options',[dict(gate_accepted=True),dict(gate_missing='FACE_LEDGER.csv'),
                                 dict(gate_missing='comparison_viewer.html'),dict(gate_missing='retained_viewer.html')])
def test_face_mask_candidate_requires_unaccepted_contract_and_real_artifacts(controller,monkeypatch,options):
    from integrations.viewer import export
    monkeypatch.setattr(export,'export_html',lambda spec,destination,profile: {})
    controller.options.update(fail=None,**options)
    result=surface.run(controller.source,controller.out)
    assert result['face_mask_gate_status']=='NOT_EVALUATED'
    assert result['preferred_viewer']==str(controller.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False
    assert result['stages'][-2]['status']=='BUSINESS_GATE_FAILED'
    assert result['stages'][-1]['name']=='shape_support'


@pytest.mark.parametrize('kind',['process','business'])
def test_support_failure_is_preserved_without_accepting_geometry(controller,monkeypatch,kind):
    from integrations.viewer import export
    monkeypatch.setattr(export,'export_html',lambda spec,destination,profile: {})
    controller.options.update(fail='surface_support',kind=kind)
    result=surface.run(controller.source,controller.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['geometry_accepted'] is False
    assert result['surface_support_status']=='NOT_EVALUATED'
    assert result['stages'][-2]['name']=='surface_support'
    assert result['stages'][-2]['status']==('FAILED' if kind=='process' else 'BUSINESS_GATE_FAILED')
    assert result['stages'][-1]['name']=='shape_support'
    assert result['error']


@pytest.mark.parametrize('placement', ['inside_source', 'outside_outputs', 'source_itself'])
def test_output_scope_rejected_before_native_access(controller, monkeypatch, placement):
    destinations = dict(inside_source=controller.source/'new-output',
                        outside_outputs=controller.root/'unapproved-output', source_itself=controller.source)
    def forbidden(path):
        raise AssertionError('native reader must not run for invalid output scope')
    monkeypatch.setitem(sys.modules, 'pycolmap', SimpleNamespace(Reconstruction=forbidden))
    with pytest.raises(ValueError, match='fresh disjoint'):
        surface.run(controller.source, destinations[placement])
    assert not controller.calls


def test_source_drift_stops_after_first_stage(controller):
    controller.options.update(fail=None, drift=True)
    result = surface.run(controller.source, controller.out)
    assert [x['name'] for x in controller.calls] == ['export']
    assert result['status'] == 'INPUT_DRIFT_REFUSED'
    assert result['binding_drift']


def test_short_view_point_is_subject_to_same_whole_point_gate():
    model = _model()
    model.points3D[7].track.elements = [SimpleNamespace(image_id=2)]
    assert surface.exclusion_candidates(model) == [7]


def test_public_surface_regions_routes_to_controller(tmp_path, monkeypatch, capsys):
    import json
    from scripts import run_photo_demo as cli

    calls = []
    source, output = tmp_path/'source', tmp_path/'fresh-output'
    result = dict(status='SURFACE_GENERATED_NOT_ACCEPTED', geometry_accepted=False)
    def run(source_run, destination, model_id):
        calls.append((source_run, destination, model_id))
        return dict(result)
    monkeypatch.setattr(surface, 'run', run)
    monkeypatch.setattr(cli, 'run_pipeline', lambda *a, **k: pytest.fail('native pipeline started'))
    assert cli.main(['surface-regions', '--run', str(source), '--output', str(output),
                     '--model-id', '3']) == 0
    assert calls == [(source, output, 3)]
    assert json.loads(capsys.readouterr().out) == result
    result['status'] = 'PARTIAL_OR_FAILED'
    assert cli.main(['surface-regions', '--run', str(source), '--output', str(output)]) == 1
    assert calls[-1] == (source, output, None)
    assert json.loads(capsys.readouterr().out) == result
