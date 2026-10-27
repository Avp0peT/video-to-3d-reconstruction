import numpy as np
import pytest
from integrations.video_masks.local_surface_adapter import seed_candidates, combine_decisions, area
from integrations.video_masks.local_subdivision import subdivide_conforming
from integrations.video_masks.local_face_evidence import prepare_probe_membership


def ledger():
    return [dict(source_face_id=str(i),decision=d,gap_conflict_views=str(v),gap_angle_pair_evaluated='True',
        gap_maximum_conflict_angle_degrees=str(a)) for i,d,v,a in [
        (0,'RETAINED_NOT_ACCEPTED',2,8),(1,'RETAINED_NOT_ACCEPTED',1,9),
        (2,'QUARANTINED_CANDIDATE',3,20),(3,'RETAINED_NOT_ACCEPTED',2,7.9)]]


def test_candidate_parent_does_not_inherit_quarantine_or_single_view():
    seed,old=seed_candidates(ledger(),[0,1,3],4)
    assert seed.tolist()==[True,False,False,False]
    assert old.tolist()==[False,False,True,False]


@pytest.mark.parametrize('damage',['order','missing','duplicates','partition'])
def test_parent_identity_failure(damage):
    rows=ledger();ids=[0,1,3]
    if damage=='order':rows.reverse()
    if damage=='missing':rows.pop()
    if damage=='duplicates':ids=[0,1,1,3]
    if damage=='partition':ids=[0,1,2,3]
    with pytest.raises(ValueError):seed_candidates(rows,ids,4)


def test_original_protected_probe_is_localized_after_two_rounds():
    v=np.array([[0.,0,2],[8,0,2],[0,8,2],[8,8,2]])
    f=np.array([[0,1,2],[1,3,2]],np.int64); parents=np.array([0,1]); eligible=np.array([True,False])
    oldv,oldf=v.copy(),f.copy()
    for _ in range(2):
        d=subdivide_conforming(v,f,parents,eligible,eligible.copy())
        v,f,parents,eligible=[d[k] for k in ('vertices','faces','parent_ids','eligible')]
    membership=prepare_probe_membership(v,f,oldv,oldf,parents)
    assert np.any(eligible & ~membership.any(axis=1))
    assert np.any((parents==1) & ~eligible) # closure never promoted
    a=np.bincount(parents,weights=area(v,f),minlength=2)
    np.testing.assert_allclose(a,area(oldv,oldf),rtol=0,atol=1e-12)
    proposed=np.ones(len(f),bool)
    q,additional=combine_decisions(parents,eligible,[False,False],proposed,membership.any(axis=1))
    assert np.array_equal(q,additional)
    assert not np.any(q & membership.any(axis=1))
    assert not np.any(q & ~eligible)


def test_baseline_quarantine_is_kept_but_new_unknown_never_votes():
    q,add=combine_decisions([0,1,2,3],[True,True,False,False],[False,False,False,True],
        [False,True,True,False],[False,True,False,True])
    assert q.tolist()==[False,False,False,True]
    assert not add.any()
