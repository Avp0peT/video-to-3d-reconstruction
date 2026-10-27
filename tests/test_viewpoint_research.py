import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pytest
from integrations.demo_pipeline import viewpoint_research as v


def test_collision_groups_transitive_and_no_coordinate_rounding():
    ids=[1,2,3,4]
    rows=[[0.,0.,1.,2.],[0.,0.,3.,4.],[6.,7.,3.,4.],[1e-12,0.,9.,10.]]
    assert v.group_pixels(ids,rows)==[[1,2,3],[4]]
    kept,excluded=v.select_injective_six(ids,rows)
    assert kept==[1,3,4]
    assert excluded==[dict(point_id=2,reason='exact_coordinate_collision',sides=[0],collision_pixels=[[0.,0.]])]


def test_split_keeps_duplicate_coordinates_together():
    ids=list(range(1,13))
    images={1:dict(observations=[(i,i*2,i+1) for i in range(12)]),
            2:dict(observations=[(i*3,i*4,i+1) for i in range(12)])}
    images[1]['observations'][1]=(0,0,2)
    points={i:dict(track={1:i-1,2:i-1},image_ids={1,2}) for i in ids}
    p=v.make_plan((1,2),images,points)
    assert [1,2] in p['duplicate_groups']
    assert 0 in p['train_indices'] and 1 in p['train_indices']
    assert not set(p['train_indices'])&set(p['holdout_indices'])
    assert sorted(p['train_indices']+p['holdout_indices'])==list(range(12))
    assert p['injective_six']==[1,3,4,5,6,7]
    assert p['raw_pixels'][1][:2]==[0,0]


def test_real_exact_synthetic_chain_and_negative(tmp_path):
    body=[(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
    a={i:(p[0],p[1]) for i,p in enumerate(body)}
    b={i:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body)}
    good=v.exact(a,b,tmp_path/'good.json')
    bad=v.exact(a,{i:b[(i+1)%6] for i in a},tmp_path/'bad.json')
    assert good['status']=='CONSTRUCTED' and good['independent']['status']=='VERIFIED'
    assert bad['status']=='INCOMPATIBLE_EXACT_PARALLEL' and bad['independent']['status']=='VERIFIED'
    assert json.loads((tmp_path/'good.json').read_text())


def test_output_guard_and_input_binding_are_checked(tmp_path):
    source=tmp_path/'missing.json'
    with pytest.raises(FileNotFoundError):v.run(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()
