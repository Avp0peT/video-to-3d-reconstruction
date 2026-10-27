from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from integrations.demo_pipeline.coverage_review import populations,build_point_coverage
from integrations.demo_pipeline.parallel_photo_audit import binding


def test_population_identity_and_distinct_native_views():
    mask=np.full((4,4),255,np.uint8);mask[3,3]=0
    image={'observations':[(.5,.5,10),(1.5,1.5,20),(2.5,2.5,-1),(3.5,3.5,30)]}
    points={10:dict(track={1:0,2:4},ambiguous_observations=False),
            20:dict(track={1:1,2:8,3:10},ambiguous_observations=False),
            30:dict(track={1:3,2:6,3:7},ambiguous_observations=False)}
    sets,counts=populations(image,points,mask,1)
    assert [r['feature_row'] for r in sets['features']]==[0,1,2]
    assert [r['point3D_id'] for r in sets['registered']]==[10,20]
    assert [r['point3D_id'] for r in sets['multiview3']]==[20]
    assert counts['outside_mask']==1
    points[20]['ambiguous_observations']=True
    assert populations(image,points,mask,1)[0]['multiview3']==[]
    points[10]['track'][1]=2
    with pytest.raises(ValueError,match='identity'):populations(image,points,mask,1)


def test_real_pixel_cover_report_preserves_feature_and_mask_distinction(tmp_path):
    mask=np.full((4,4),255,np.uint8);gray=np.full((4,4),100,np.uint8)
    mp=tmp_path/'mask.png';gp=tmp_path/'gray.png';Image.fromarray(mask).save(mp);Image.fromarray(gray).save(gp)
    source={'images':{1:{'observations':[(.5,.5,10),(1.5,1.5,-1)]}},
        'points':{10:dict(track={1:0,2:2,3:3},ambiguous_observations=False)},
        'frames':{1:dict(input_index=7,binding=binding(gp))},'masks':{1:mask},'gray':{1:gray},
        'mask_info':{1:dict(binding=binding(mp))}}
    before=(gp.read_bytes(),mp.read_bytes());out=tmp_path/'out';result=build_point_coverage(source,out)
    assert result['status']=='COMPLETE'
    record=json.loads((out/'RESULT.json').read_text(encoding='utf-8'));frame=record['frames'][0]
    for radius in frame['radii']:
        data=radius['sets']['features']
        assert data['cover']['complete'] and data['cover']['selected_feature_rows']==[0,1]
        assert data['all_points_mask_coverage']['covered_pixels']==2
        assert data['all_points_mask_coverage']['uncovered_pixels']==14
        ref=data['all_points_mask_coverage']['map'];assert ref['readback']
        with Image.open(ref['path']) as im:assert np.count_nonzero(np.asarray(im)==1)==2
    assert record['geometry_changed'] is False and (gp.read_bytes(),mp.read_bytes())==before
    with pytest.raises(FileExistsError):build_point_coverage(source,out)


def test_missing_mask_is_unavailable_not_empty_success(tmp_path):
    source={'images':{1:{}},'frames':{1:dict(input_index=1)},'masks':{}}
    result=build_point_coverage(source,tmp_path/'out')
    assert result['status']=='PARTIAL' and result['evaluated_frames']==0
