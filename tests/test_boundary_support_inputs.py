import copy
import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import review_boundary_support as w


def protocol():
    validated={};local={};frozen=[];old=[];association=[]
    for lo,hi in w.CLIPS:
        for i in range(lo,hi+1):
            status='NEEDS_SELECTION' if i in (81,84,85,86) else ('NO_MATCH' if i in range(52,58) else 'OK')
            f=dict(input_index=i,frame_id=f'f{i}',original_mask_status=status)
            validated[i]=dict(frame=f,candidates=[])
        frames={i:validated[i]['frame'] for i in range(lo,hi+1)}
        for i in range(lo,hi+1):
            f=frames[i];pair=w.reference_pair(frames,i)
            row=dict(input_index=i,frame_id=f['frame_id'],original_status=f['original_mask_status'],before_index=pair[0] if pair else None,after_index=pair[1] if pair else None,track_ids=[],geometry_segments=[],points=dict(query=[]),selected_candidate_id=None,publishable_for_geometry=False)
            local[i]=row;old.append(copy.deepcopy(row))
            frozen.append(dict(input_index=i,original_frame=f,track_ids=[],geometry_segments=[],query_points_xy=[]))
            association.append(dict(**f,candidate_bindings=[]))
    return validated,dict(frames=frozen),local,dict(frames=association),old


def test_prepare_groups_do_not_duplicate_pairs():
    data=protocol();prepared,groups=w.prepare_rows(*data)
    assert len(prepared)==28 and groups==[(80,82),(83,87)]
    assert sum(p['exterior_pair']==(83,87) for p in prepared)==3


@pytest.mark.parametrize('kind',['bool','frameid','originalstatus','pair','positive'])
def test_prepare_rejects_identity_mismatch(kind):
    data=protocol();local=data[2]
    if kind=='bool':local[81]['input_index']=True
    elif kind=='frameid':local[81]['frame_id']='different'
    elif kind=='originalstatus':local[81]['original_status']='OK'
    elif kind=='pair':local[81]['before_index']=79
    elif kind=='positive':local[81]['local_motion']=dict(profiles=dict(local_affine=dict(measured_query_evidence=[dict(track_id=1,query_point_xy=[1,2])],common_supported_track_ids=[1])))
    with pytest.raises(ValueError):w.prepare_rows(*data)


def test_query_actual_observation_and_conditional_label():
    ledger=dict(accepted_witnesses=[dict(witness_id=3,observations=[dict(frame_index=i,xy=[i+.25,9]) for i in range(80,83)])])
    result=w.query_witnesses(ledger,(80,82),81)
    assert result[0]['witness_id']=='80_82:3'
    assert result[0]['query_point_xy']==[81.25,9]
    assert result[0]['verified_background'] is False


@pytest.mark.parametrize('case',['missing','boolindex','duplicate','boolid'])
def test_query_rejects_missing_or_invalid(case):
    ledger=dict(accepted_witnesses=[dict(witness_id=3,observations=[dict(frame_index=i,xy=[9,9]) for i in range(80,83)])])
    if case=='missing':ledger['accepted_witnesses'][0]['observations'].pop(1)
    elif case=='boolindex':ledger['accepted_witnesses'][0]['observations'][1]['frame_index']=True
    elif case=='duplicate':ledger['accepted_witnesses']*=2
    else:ledger['accepted_witnesses'][0]['witness_id']=True
    with pytest.raises(ValueError):w.query_witnesses(ledger,(80,82),81)


def test_fresh_no_overwrite_or_overlap(tmp_path):
    source=tmp_path/'source';source.mkdir()
    sentinel=source/'sentinel';sentinel.write_text('keep')
    for output in (source,source/'child',tmp_path):
        with pytest.raises(ValueError):w.fresh_output(output,[source])
    assert sentinel.read_text()=='keep'
    assert w.fresh_output(tmp_path/'new',[source])==tmp_path/'new'


def test_stale_local_seal_is_rejected_before_read(tmp_path):
    local=tmp_path/'task/real01/local_motion';local.mkdir(parents=True)
    (tmp_path/'task/SHA256SUMS.txt').write_text('stale')
    with pytest.raises(ValueError,match='binding drift'):w.sealed_local_review(local)


def test_countcv_records_actual_calls():
    class Fake:
        def calcOpticalFlowPyrLK(self,*a,**k):return 'actual'
    cv=w.CountCV(Fake())
    assert cv.calcOpticalFlowPyrLK()==cv.calcOpticalFlowPyrLK()=='actual'
    assert cv.flow_calls==2
