import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from integrations.demo_pipeline.reassociation_report import render_reassociation


def data(count=1):
    frames=[dict(input_index=i,time_seconds=i/10,model_ids=[0],original_mask_status='已预测<script>') for i in range(3)]
    grays=[np.arange(100*120,dtype=np.uint8).reshape(100,120)+i for i in range(3)]
    links=[dict(from_frame=2,to_frame=0,from_track_id=i,to_track_id=i+10,direction='backward<script>',
                xy_from=[50.5,40.5],xy_to=[55.5,45.5],missing_frame_indices=[1],metrics={'注释':'<script>alert(1)</script>'},
                status='CANDIDATE_NOT_IDENTITY_VERIFIED') for i in range(count)]
    recovery=dict(links=links,joined_tracks=[dict(original_observations=[{'frame':2,'xy':[50.5,40.5]}],sourceIDs=[1,2])],baseline_continuity={'continuous_links':0})
    return recovery,dict(counts={'candidate':count},per_link_results=[]),frames,grays


def test_empty_is_usable_diagnostic(tmp_path):
    r,a,f,g=data(0)
    result=render_reassociation(tmp_path/'empty',r,a,f,g)
    assert result['status']=='NO_CANDIDATES' and result['displayed_count']==0
    assert '没有可展示的候选' in (tmp_path/'empty'/'index.html').read_text(encoding='utf-8')
    assert not list((tmp_path/'empty').glob('*.png'))


def test_zero_candidates_still_exposes_attempt_rejections(tmp_path):
    r,a,f,g=data(0)
    r.update(counts={'endpoint_comparisons':17,'accepted_links':0},
             rejection_counts={'OUTSIDE_MASK':12,'FORWARD_BACKWARD_REJECTED':5},
             profile={'max_gap':3},proposal_summary={'proposed_links':0},patch_comparisons=0)
    result=render_reassociation(tmp_path/'reasons',r,a,f,g)
    stored=json.loads((tmp_path/'reasons'/'REPORT.json').read_text(encoding='utf-8'))
    for key in ('counts','rejection_counts','profile','proposal_summary','patch_comparisons'):
        assert result[key]==stored[key]==r[key]
    page=(tmp_path/'reasons'/'index.html').read_text(encoding='utf-8')
    assert 'OUTSIDE_MASK' in page and 'FORWARD_BACKWARD_REJECTED' in page
    assert '不是互斥的物理点计数' in page and 'patch_comparisons=0：纹理检查未到达' in page


def test_backward_coordinates_original_pixels_and_no_mutation(tmp_path):
    r,a,f,g=data()
    snapshot=copy.deepcopy((r,a,f)); pixels=[x.copy() for x in g]
    output=tmp_path/'out'
    result=render_reassociation(output,r,a,f,g)
    assert (r,a,f)==snapshot
    assert all(np.array_equal(x,y) for x,y in zip(g,pixels))
    panel=result['panels'][0]
    assert [e['input_index'] for e in panel['endpoints']]==[2,0]
    assert panel['endpoints'][0]['xy_zero_based_pixel']==[50.,40.]
    assert panel['endpoints'][0]['crop_box_xyxy']==[10,0,91,81]
    assert panel['missing_frame_indices']==[1] and panel['missing_observations_status']=='UNKNOWN_NOT_FILLED'
    assert not result['missing_frames_filled'] and not result['sfm_injected']
    with Image.open(output/panel['file']) as image:
        actual=np.array(image)
        assert np.array_equal(actual[:81,:81],g[2][0:81,10:91])
        assert hashlib.sha256(image.tobytes()).hexdigest()==panel['png_pixel_sha256']
    assert json.loads((output/'REPORT.json').read_text(encoding='utf-8'))['candidate_count']==1
    page=(output/'index.html').read_text(encoding='utf-8')
    assert '<script>' not in page and '&lt;script&gt;' in page
    with pytest.raises(ValueError,match='fresh'):
        render_reassociation(output,r,a,f,g)


def test_uniform_bounded_not_quality_selection(tmp_path):
    r,a,f,g=data(11)
    result=render_reassociation(tmp_path/'out',r,a,f,g)
    # Ordering is by actual endpoint track identities as strings, followed by uniform spread.
    order=sorted(range(11),key=lambda i:str(i))
    assert [p['link_index'] for p in result['panels']]==[order[i] for i in (0,2,4,6,8,10)]
    assert len(list((tmp_path/'out').glob('*.png')))==6


def test_sequence_indices_bind_sparse_original_input_indices(tmp_path):
    r,a,f,g=data()
    for frame,index in zip(f,[7,29,51]):
        frame['input_index']=index
    panel=render_reassociation(tmp_path/'out',r,a,f,g)['panels'][0]
    assert [e['frame_index'] for e in panel['endpoints']]==[2,0]
    assert [e['input_index'] for e in panel['endpoints']]==[51,7]
    assert panel['missing_input_indices']==[29]


@pytest.mark.parametrize('xy',[[float('nan'),1],[0,1],[120,1],[1]])
def test_does_not_invent_invalid_endpoint(tmp_path,xy):
    r,a,f,g=data();r['links'][0]['xy_from']=xy
    with pytest.raises(ValueError):
        render_reassociation(tmp_path/'refused',r,a,f,g)
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('field,value',[
    ('from_frame',-1),('to_frame',3),('from_frame',True),('to_frame',1.0),
    ('missing_frame_indices',[-1]),('missing_frame_indices',[True]),
    ('missing_frame_indices',[1,1]),('missing_frame_indices',[]),
    ('missing_frame_indices',[0]),('missing_frame_indices',[3]),
])
def test_frame_indices_rejected_before_output(tmp_path,field,value):
    r,a,f,g=data();r['links'][0][field]=value
    with pytest.raises(ValueError,match='frame ind'):
        render_reassociation(tmp_path/'refused',r,a,f,g)
    assert not (tmp_path/'refused').exists()


def test_forward_missing_interval_and_backward_order(tmp_path):
    r,a,f,g=data()
    r['links'][0].update(from_frame=0,to_frame=2,direction='forward',missing_frame_indices=[1])
    assert render_reassociation(tmp_path/'forward',r,a,f,g)['panels'][0]['missing_frame_indices']==[1]
    # Four frames make the backward ordering requirement distinguishable.
    f.append(dict(f[-1],input_index=3));g.append(g[-1].copy())
    r['links'][0].update(from_frame=3,to_frame=0,missing_frame_indices=[1,2])
    with pytest.raises(ValueError,match='ordered open'):
        render_reassociation(tmp_path/'wrong_order',r,a,f,g)
    assert not (tmp_path/'wrong_order').exists()
    r['links'][0]['missing_frame_indices']=[2,1]
    assert render_reassociation(tmp_path/'backward',r,a,f,g)['panels'][0]['missing_frame_indices']==[2,1]
