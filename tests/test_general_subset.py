"""Offline native restriction fixtures, no reconstruction or GPU."""
import json
import numpy as np
import pytest
from test_native_subset import native, chain43, sha
from integrations.mvs_surface import general_subset as g


def test_relation_nonconsecutive_ids_and_whole_duplicate_deletion(tmp_path):
    original=native([4,17,60],duplicates=1,count=3,observed=2)
    path=tmp_path/'source';path.mkdir();original.write_binary(str(path))
    child=g._native(path)
    child.delete_point3D(4);child.deregister_frame(60)
    relation=g.compare(original,child,[4,17])
    assert relation['duplicate_image_points']==[4]
    assert relation['derived_registered_ids']==[4,17]
    assert len(relation['removed_observations'])==5
    child.point3D(1).xyz=np.array([99.,1.,5.])
    with pytest.raises(ValueError):g.compare(original,child,[4,17])


@pytest.mark.parametrize("contract", ["sfm-native-object-v1", "sfm-native-object-pinhole-v1"])
def test_derive_general_and_independent_certificate(chain43,tmp_path,contract):
    c=chain43
    path=c["sfm"]/"run.json"
    doc=json.loads(path.read_text());doc["contract_version"]=contract;path.write_text(json.dumps(doc))
    # Arbitrary explicit exclusion, independent of historical native60 profile.
    from test_native_subset import INDICES
    indices=[i for i in INDICES if i not in (59,60)]
    source_before=[g._binding(c['sfm']/'models/0/binary'/f) for f in g.MODEL_BIN]
    out=g.derive(c['sfm'],c['t01'],'0',indices,tmp_path/'general',{59:'NO_MATCH',60:'NOT_SELECTED'})
    cert=g.validate_reference(out['subset_certificate'])
    assert {59,60} <= set(cert['removed_indices'])
    assert len(cert['removed_indices'])==66-len(indices)
    assert cert['relation']['duplicate_image_points'] and len(cert['retained_native_ids'])==42
    assert all(g._binding(b['path'])==b for b in source_before)
    p=out['subset_certificate']['path'];doc=json.loads(open(p,encoding='utf-8').read())
    doc['relation']['derived_points']+=1
    from pathlib import Path
    Path(p).write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(ValueError,match='independently recomputed'):
        g.validate_reference({'path':p,'sha256':sha(Path(p))})


def test_general_refuses_too_few_or_unknown_views(chain43,tmp_path):
    c=chain43
    with pytest.raises(ValueError):g.derive(c['sfm'],c['t01'],'0',[3],tmp_path/'short')
    with pytest.raises(ValueError):g.derive(c['sfm'],c['t01'],'0',[0,3],tmp_path/'unknown')


def test_deregister_removes_short_tracks_and_multiple_exclusions(tmp_path):
    original=native([4,17,60,91],count=3,observed=1)
    path=tmp_path/'source';path.mkdir();original.write_binary(str(path))
    child=g._native(path)
    child.deregister_frame(17);child.deregister_frame(91)
    assert child.num_points3D()==1
    result=g.compare(original,child,[4,60])
    assert result['removed_points']==[2,3]
    assert result['derived_points']==1


def test_all_selected_still_removes_whole_duplicate_points(chain43,tmp_path):
    from test_native_subset import INDICES
    c=chain43;out=g.derive(c['sfm'],c['t01'],'0',INDICES,tmp_path/'all')
    cert=g.validate_reference(out['subset_certificate'])
    assert len(cert['retained_native_ids'])==44
    assert len(cert['relation']['duplicate_image_points'])==20


def test_partial_alias_selection_refused(chain43,tmp_path):
    from test_native_subset import INDICES
    c=chain43;path=c['sfm']/'frames_map.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows[0]=dict(rows[3],input_index=0)
    # A claimed alias also has to bind matching canonical IO; build byte-identical alias.
    frames_path=c['t01']/'frames.jsonl';frames=[json.loads(x) for x in frames_path.read_text().splitlines()]
    frames[0]=dict(frames[3],input_index=0)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    frames_path.write_text(''.join(json.dumps(x)+'\n' for x in frames),encoding='utf-8')
    with pytest.raises(ValueError,match='partially selected alias'):
        g.derive(c['sfm'],c['t01'],'0',INDICES,tmp_path/'alias')


@pytest.mark.parametrize('fault', [None, 'contract', 'frame_identity', 'source_hash'])
def test_video192_general_certificate_keeps_missing_views_and_rejects_drift(chain43, tmp_path, fault):
    from test_native_subset import INDICES
    c = chain43
    frames_path, maps_path = c['t01']/'frames.jsonl', c['sfm']/'frames_map.jsonl'
    frames = [json.loads(s) for s in frames_path.read_text().splitlines()]
    maps = [json.loads(s) for s in maps_path.read_text().splitlines()]
    for i in range(len(frames), 192):
        frames.append(dict(frames[0], input_index=i, frame_id=f'TEST_DATA_missing_{i}'))
        maps.append(dict(maps[0], input_index=i, frame_id=f'TEST_DATA_missing_{i}', registrations=[],
                         staged_image_name=None, representative_input_index=None))
    for path, rows in ((frames_path, frames), (maps_path, maps)):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    for path in (c['t01']/'run.json', c['sfm']/'run.json'):
        data = json.loads(path.read_text())
        if path.parent == c['t01']:
            data.update(input_count=192, ok_count=192, error_count=0)
        else:
            data.update(contract_version='sfm-native-object-video-v1', profile_id='cad-object-video-v1',
                        counts={'input_rows':192})
            if fault == 'contract': data['contract_version'] = 'sfm-native-object-video-v999'
        path.write_text(json.dumps(data), encoding='utf-8')
    if fault == 'frame_identity':
        maps[3]['frame_id'] = 'WRONG'
        maps_path.write_text(''.join(json.dumps(r)+'\n' for r in maps), encoding='utf-8')
    if fault == 'source_hash':
        maps[3]['canonical_image_sha256'] = '0'*64
        maps_path.write_text(''.join(json.dumps(r)+'\n' for r in maps), encoding='utf-8')
    indices = [i for i in INDICES if i not in (59, 60)]
    if fault:
        with pytest.raises(ValueError):
            g.derive(c['sfm'], c['t01'], '0', indices, tmp_path/'video')
    else:
        result = g.derive(c['sfm'], c['t01'], '0', indices, tmp_path/'video')
        cert = g.validate_reference(result['subset_certificate'])
        assert len(cert['selected_inputs']) == len(indices)
        assert len(cert['excluded_inputs']) == 192-len(indices)
        assert set(range(66,192)) <= set(cert['removed_indices'])
