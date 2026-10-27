import json
import sqlite3
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import post_review as p


def test_missing_model_generates_report_and_does_not_touch_pipeline(tmp_path):
    source=tmp_path/'pipeline';source.mkdir()
    data={'stages':{'sfm':{'status':'NO_MODEL'}}}
    text=json.dumps(data);(source/'pipeline.json').write_text(text)
    out=tmp_path/'review';result=p.review(source,out)
    assert result['status']=='INSUFFICIENT_DATA'
    assert (out/'index.html').exists() and json.loads((out/'DISCOVERY.json').read_text())==data
    assert (source/'pipeline.json').read_text()==text
    with pytest.raises(ValueError,match='new review'):p.review(source,out)


def test_view_selection_is_bounded_and_includes_endpoints():
    assert p.evenly_spaced(list(range(5)))==list(range(5))
    selected=p.evenly_spaced(list(range(44)))
    assert len(set(selected))==12 and selected[0]==0 and selected[-1]==43


def test_local_control_no_zero_displacement_or_residual_selection():
    plan={'raw_pixels':[[0,0,1,1],[9,9,1,1],[2,2,2,1],[7,6,6,8]],
          'point_ids':[10,11,12,13],'holdout_indices':[0,1,2,3],'image_ids':[1,2]}
    swap,record=p.closest_swap(plan)
    assert swap['source_point_ids']==[10,12] and record['displacement_pixels']==1


def database(path,xy):
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE images(image_id,name,camera_id); CREATE TABLE keypoints(image_id,rows,cols,data); CREATE TABLE descriptors(image_id,type,rows,cols,data);')
    db.execute('INSERT INTO images VALUES(1,?,1)',('images/a.png',))
    db.execute('INSERT INTO keypoints VALUES(1,2,2,?)',(np.array(xy,dtype='<f4').tobytes(),))
    db.execute('INSERT INTO descriptors VALUES(1,0,2,128,?)',(bytes(range(128))*2,));db.commit();db.close()


def test_feature_identity_full_native_readback_and_readonly(tmp_path):
    path=tmp_path/'database.db';database(path,[[1,2],[3,4]])
    image={1:dict(name='images/a.png',camera_id=1,observations=[(1,2,10),(3,4,11)])}
    before=path.read_bytes();descriptors,info=p.descriptor_database(path,image)
    assert descriptors[1].shape==(2,128) and info['status']=='AVAILABLE'
    assert path.read_bytes()==before
    image[1]['observations'][1]=(5,4,11)
    with pytest.raises(ValueError,match='keypoint order'):p.descriptor_database(path,image)


def test_appearance_summary_keeps_unavailable_and_two_radii_separate():
    row={'descriptor':dict(status='EVALUATED',original_distance=.2,alternative_distance=.4,support='original'),
         'patches':{'8':dict(status='EVALUATED',original_zncc=.1,alternative_zncc=.5,support='alternative'),
                    '16':dict(status='NOT_EVALUATED_MASK_BOUNDARY',support=None)}}
    r=p.summarize_appearance([row])
    assert r['descriptor']['support']=='original' and r['patch8']['support']=='alternative'
    assert r['patch16']['status']=='UNAVAILABLE' and r['patch16']['unavailable_count']==1


@pytest.mark.parametrize('role',['pipeline','sfm','segmentation','photos'])
def test_source_subdirectory_is_rejected_before_creation(tmp_path,role):
    pipeline=tmp_path/'pipeline';pipeline.mkdir()
    roots={k:tmp_path/k for k in ('sfm','segmentation','photos')}
    discovery={'source':str(roots['photos']),'stages':{
        'sfm':{'run_json':str(roots['sfm']/'run.json')},'segmentation':{'run_dir':str(roots['segmentation'])}}}
    (pipeline/'pipeline.json').write_text(json.dumps(discovery))
    output=(pipeline/'input' if role=='pipeline' else roots[role])/'new_review'
    with pytest.raises(ValueError,match='overlaps'):p.review(pipeline,output)
    assert not output.exists()


def test_exact_automatic_child_is_allowed(tmp_path):
    pipeline=tmp_path/'pipeline';pipeline.mkdir()
    (pipeline/'pipeline.json').write_text('{"stages":{}}')
    result=p.review(pipeline,pipeline/'post_review')
    assert result['status']=='INSUFFICIENT_DATA' and (pipeline/'post_review/index.html').exists()


def test_omitted_view_cannot_hide_candidate_alias():
    full={pid:{i:[float(pid+i),float(pid-i)] for i in range(1,14)} for pid in (1,2,3)}
    selected=p.evenly_spaced(list(range(1,14)))
    assert 12 not in selected
    cut=lambda:{pid:{i:xy for i,xy in views.items() if i in selected} for pid,views in full.items()}
    assert p.aliases_preserved(full,cut())
    full[3][12]=full[1][12][:]
    assert not p.aliases_preserved(full,cut())
    full[3][1]=full[1][1][:]
    assert p.aliases_preserved(full,cut())


def test_zero_evidence_cannot_pass_by_having_cases():
    case={k:{'status':'UNAVAILABLE'} for k in ('pair','extra_summary','pair_plus_extra_summary')}
    case['appearance_summary']={k:{'evaluated_count':0} for k in ('descriptor','patch8','patch16')}
    coverage,available=p.evidence_coverage([case])
    assert not available and coverage['extra_summary']==0
    case['appearance_summary']['patch8']['evaluated_count']=1
    assert p.evidence_coverage([case])[1]
