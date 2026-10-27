"""TEST_DATA: continuation never replays failed native stages or hides drift."""
import pytest
from integrations.video_masks import region_improvement as imp


@pytest.mark.parametrize('failure,expected', [
    (None, ['identity','tracking','geometry','assessment']),
    ('tracking', ['identity','tracking']),
    ('geometry', ['identity','tracking','geometry'])])
def test_stage_order_stop_and_one_shot(tmp_path, monkeypatch, failure, expected):
    imp.save(tmp_path/'PLAN.json', dict(source_bindings={}, code_bindings={}))
    imp.save(tmp_path/'SUMMARY.json', dict(status='PREPARED', stages={}))
    seen=[]
    def invoke(root, run, stage, argv, **kw):
        seen.append(stage)
        assert kw['memory'] == 4 << 30
        if stage == 'assessment' and failure != stage:
            (run/'assessment').mkdir()
            imp.save(run/'assessment'/'ASSESSMENT.json', {'geometry_accepted':False})
        return dict(status='completed', root_exit_code=1 if failure == stage else 0,
                    reclaim_confirmed=True, errors=[], log_errors=[])
    monkeypatch.setattr(imp, '_invoke', invoke)
    result=imp.execute(tmp_path)
    assert seen == expected
    assert result['status'] == ('PARTIAL_DIAGNOSTIC' if failure else 'EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED')
    with pytest.raises(ValueError, match='already attempted'):
        imp.execute(tmp_path)


def test_prelaunch_code_drift_refuses(tmp_path):
    code=tmp_path/'bound.txt';code.write_text('original')
    imp.save(tmp_path/'PLAN.json',dict(source_bindings={},code_bindings={str(code):imp.sha(code)}))
    imp.save(tmp_path/'SUMMARY.json',dict(status='PREPARED',stages={}))
    code.write_text('changed')
    with pytest.raises(ValueError,match='changed after preparation'):
        imp.execute(tmp_path)
    assert not (tmp_path/'logs').exists()


def test_source_model_mutation_after_prepare_refuses_before_launch(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir()
    geometry=source/'other_geometry';geometry.mkdir()
    binary=geometry/'binary';binary.mkdir()
    text=geometry/'text';text.mkdir()
    model=binary/'points3D.bin';model.write_bytes(b'bound original model')
    (text/'points3D.txt').write_text('bound text')
    ply=geometry/'points.ply';ply.write_text('bound ply')
    (geometry/'database.db').write_bytes(b'bound db')
    imp.save(geometry/'PLAN.json', {})
    imp.save(geometry/'RESULT.json', {'models':[dict(binary_path=str(binary),text_path=str(text),
        ply_path=str(ply),registered_views=2,points=3,model_id=7)]})
    imp.save(source/'PLAN.json', dict(source_bindings={},baseline_sfm='unused'))
    imp.save(source/'COMPARISON.json', {})
    (source/'tracking').mkdir()
    imp.save(source/'tracking'/'REGION_TRACKS.json',dict(schema='glove-region-tracks-v1',source_bindings=[]))
    monkeypatch.setattr(imp,'CODE',())
    output=tmp_path/'new'
    plan=imp.prepare(source,output,'other_geometry')
    assert plan['source_identity_model']==7
    assert str(model) in plan['source_bindings']
    model.write_bytes(b'replaced model')
    with pytest.raises(ValueError,match='changed after preparation'):
        imp.execute(output)
    assert not (output/'logs').exists()


def _source_fixture(tmp_path, monkeypatch, geometry_names):
    source = tmp_path / 'source'
    source.mkdir()
    imp.save(source / 'PLAN.json', dict(source_bindings={}, baseline_sfm='TEST_DATA'))
    imp.save(source / 'COMPARISON.json', {})
    (source / 'tracking').mkdir()
    imp.save(source / 'tracking' / 'REGION_TRACKS.json',
             dict(schema='glove-region-tracks-v1', source_bindings=[]))
    for name in geometry_names:
        geometry = source / name
        geometry.mkdir(parents=True)
        (geometry / 'database.db').write_bytes(b'TEST_DATA database')
        model = geometry / 'model.bin'
        model.write_bytes(b'TEST_DATA model')
        imp.save(geometry / 'PLAN.json', {})
        imp.save(geometry / 'RESULT.json', dict(models=[dict(binary_path=str(model),
            text_path=str(model), ply_path=str(model), registered_views=2, points=3, model_id=0)]))
    monkeypatch.setattr(imp, 'CODE', ())
    return source


@pytest.mark.parametrize('name', ['geometry', 'geometry02'])
def test_default_discovers_only_geometry_and_binds_it(tmp_path, monkeypatch, name):
    source = _source_fixture(tmp_path, monkeypatch, [name])
    plan = imp.prepare(source, tmp_path / 'output')
    assert plan['source_geometry'] == name
    assert str(source / name / 'RESULT.json') in plan['source_bindings']
    assert plan['schema'] == 'glove-region-improvement-v1'
    assert 'object-agnostic' in plan['object_scope']
    assert plan['budgets']['geometry_seconds'] == 600


def test_two_geometries_refuse_guess_and_require_explicit_selection(tmp_path, monkeypatch):
    source = _source_fixture(tmp_path, monkeypatch, ['geometry', 'geometry02'])
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='ambiguous'):
        imp.prepare(source, output)
    assert not output.exists()
    plan = imp.prepare(source, output, source / 'geometry')
    assert plan['source_geometry'] == 'geometry'


@pytest.mark.parametrize('selection', ['../source/geometry', 'geometry/../geometry', '.', 'geometry/RESULT.json'])
def test_explicit_geometry_must_be_contained_directory_without_parent_segments(tmp_path, monkeypatch, selection):
    source = _source_fixture(tmp_path, monkeypatch, ['geometry'])
    with pytest.raises(ValueError, match='source geometry'):
        imp.prepare(source, tmp_path / 'output', selection)
    assert not (tmp_path / 'output').exists()


def test_explicit_geometry_outside_source_refused(tmp_path, monkeypatch):
    source = _source_fixture(tmp_path, monkeypatch, ['geometry'])
    external = tmp_path / 'outside'
    external.mkdir()
    imp.save(external / 'RESULT.json', {})
    with pytest.raises(ValueError, match='inside source'):
        imp.prepare(source, tmp_path / 'output', external)


def test_improvement_cli_passes_none_for_unspecified_geometry(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(imp, 'prepare', lambda source, output, geometry: selected.append(geometry))
    assert imp.main(['prepare', '--source-run', str(tmp_path), '--run', str(tmp_path / 'new')]) == 0
    assert selected == [None]
