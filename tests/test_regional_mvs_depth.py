import numpy as np
from integrations.mvs_surface.region_dense_experiment import compare_depth,leakage,union_counts


def test_common_only_and_unknown_denominators():
    w=np.array([[1.,1.,0.,0.]]);r=np.array([[1.01,0.,2.,0.]]);m=np.ones(w.shape,bool)
    a=compare_depth(w,r,m)
    assert [a[k] for k in ('common_valid','agree_2pct','only_whole','only_region','both_unknown')]==[1,1,1,1,1]
    assert a['agree_fraction']==1.
    assert compare_depth(w,r,m,False)['status']=='UNKNOWN_NO_SEED'
    assert compare_depth(None,r,m)['common_valid'] is None
    assert compare_depth(None,r,m)['status']=='UNKNOWN_MISSING_DEPTH'


def test_disagreement_not_hidden_by_coverage():
    a=compare_depth(np.ones((2,2)),np.full((2,2),2.),np.ones((2,2),bool))
    assert a['regional_valid']==4 and a['agree_2pct']==0 and a['disagree_2pct']==4


def test_leakage_and_empty_seed_are_different():
    m=np.zeros((20,20),bool);m[4:8,4:8]=True
    d=np.zeros(m.shape);d[6,6]=1
    assert leakage(d,m)['positive_leak_pixels']==0
    d[16,16]=1;assert leakage(d,m)['positive_leak_pixels']==1
    assert leakage(None,m)['positive_leak_pixels'] is None
    assert leakage(np.zeros(m.shape),np.zeros_like(m))['positive_leak_pixels']==0


def test_union_keeps_conflicts_and_missing_groups_without_averaging():
    m=np.ones((5,5),bool);a=np.ones(m.shape);b=np.full(m.shape,2.)
    r=union_counts([a,b,None],[m,m,m],m)
    assert r['union_valid_pixels']==25 and r['multiple_group_valid_pixels']==25
    assert r['overlap_conflict_pixels']==25 and r['missing_groups']==[2]
    assert union_counts([None]*3,[m]*3,m)['union_valid_pixels'] is None


def test_disjoint_group_domains_do_not_create_overlap_conflict():
    m=np.ones((4,4),bool);left=np.zeros_like(m);left[:,:2]=True
    r=union_counts([np.ones(m.shape),np.full(m.shape,2.),None],[left,~left,m],m)
    assert r['union_valid_pixels']==16 and r['multiple_group_valid_pixels']==0 and r['overlap_conflict_pixels']==0
