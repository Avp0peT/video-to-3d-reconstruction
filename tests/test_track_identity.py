"""Synthetic ambiguity counterexamples, real CPU LK and lifecycle tests."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import region_tracks as rt
from integrations.video_masks.track_identity import default_policy, conflict_pairs, isolate_targets, require_unique_targets


@pytest.mark.parametrize('xy',[
    [[30.,40.],[30.,40.],[90.,80.]],
    [[30.,40.],[30.5,40.],[90.,80.]],
    [[30.,40.],[31.,40.],[90.,80.]],
])
def test_competing_ids_both_stop_without_moving_or_merging(xy):
    original=np.array(xy)
    assert isolate_targets(original,['accepted']*3,default_policy())==['target_identity_ambiguous']*2+['accepted']
    np.testing.assert_array_equal(original,xy)
    with pytest.raises(ValueError,match='unresolved'):require_unique_targets(original,default_policy())
    assert isolate_targets(original[[2,1,0]],['accepted']*3,default_policy())==['accepted']+['target_identity_ambiguous']*2


def test_nearby_resolved_points_and_failed_proposal_do_not_steal_identity():
    assert isolate_targets([[10,10],[11.01,10]],['accepted']*2,default_policy())==['accepted']*2
    assert isolate_targets([[10,10],[10,10]],['accepted','patch_zncc'],default_policy())==['accepted','patch_zncc']
    require_unique_targets([[10,10],[12,10]],default_policy())
    assert conflict_pairs([],default_policy())==[]


def test_conflict_chain_not_score_or_order_winner():
    assert isolate_targets([[10,10],[10.8,10],[11.6,10]],['accepted']*3,default_policy())==['target_identity_ambiguous']*3


@pytest.mark.parametrize('bad',[[[float('nan'),1]],[[0,0]]*601,[[1,2,3]]])
def test_invalid_targets_rejected(bad):
    with pytest.raises(ValueError):conflict_pairs(bad,default_policy())


def test_no_bool_radius_or_unfrozen_policy():
    for field,value in [('target_separation_px',True),('target_separation_px',2.),('max_active',True),('merge_ids',0)]:
        p=default_policy();p[field]=value
        with pytest.raises(ValueError,match='policy'):conflict_pairs([],p)


@pytest.fixture(scope='module')
def cv():
    result=rt.load_cv([])
    assert result is not None,'CPU OpenCV required for this regression'
    result.setNumThreads(2)
    return result


def test_real_lk_ambiguity_and_nonconflicting_control(cv):
    gray=cv.GaussianBlur(np.random.default_rng(48).integers(30,170,(128,128),dtype=np.uint8),(3,3),.7)
    mask=np.full_like(gray,255)
    p=[[40,40],[40.4,40],[80,80]]
    q,old,scores=rt.step(gray,gray,mask,mask,p,cv)
    assert old==['accepted']*3 and min(scores)>.99
    newq,new,_=rt.step(gray,gray,mask,mask,p,cv,default_policy())
    assert new==['target_identity_ambiguous']*2+['accepted']
    np.testing.assert_allclose(newq,q,atol=1e-5)
    assert rt.step(gray,gray,mask,mask,[[40,40],[48,40],[80,80]],cv,default_policy())[1]==['accepted']*3


def test_run_ends_old_ids_records_ambiguity_then_new_birth(tmp_path,cv,monkeypatch):
    rows=[]
    gray=np.random.default_rng(7).integers(30,200,(128,128),dtype=np.uint8)
    for i in range(3):
        row=dict(decode_index=i,input_index=i,width=128,height=128)
        for kind,array in [('image',gray),('mask',np.full_like(gray,255))]:
            path=tmp_path/f'{kind}{i}.png';Image.fromarray(array).save(path)
            row[kind+'_path']=str(path);row[kind+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    calls=[]
    def seeds(a,b,existing,c):
        calls.append(1)
        return [np.array(p,dtype=np.float32) for p in ([[40,40],[48,40]] if len(calls)==1 else ([[44,40]] if not existing else []))]
    def step(a,b,c,d,points,cv,policy):
        q=np.array([[44,40],[44.1,40]],np.float32) if len(points)==2 else np.array(points)
        return q,isolate_targets(q,['accepted']*len(q),policy),[1.]*len(q)
    monkeypatch.setattr(rt,'seed',seeds);monkeypatch.setattr(rt,'step',step)
    path=tmp_path/'PLAN.json'
    path.write_text(json.dumps(dict(schema='glove-region-plan-v1',timeline=rows,identity_policy=default_policy())))
    r=rt.run(path,tmp_path/'out')
    assert [o['track_id'] for o in r['frames'][0]['observations']]==[0,1]
    assert [o['track_id'] for o in r['frames'][1]['observations']]==[2]
    assert [o['track_id'] for o in r['frames'][2]['observations']]==[2]
    assert len(r['timeline_stats'][1]['identity_conflicts'])==2
    assert all(x['action']=='END_ID_NO_OBSERVATION' for x in r['timeline_stats'][1]['identity_conflicts'])
    assert all(len(t['observations'])==1 for t in r['tracks'][:2])
    assert r['identity_policy']==default_policy()
