"""TEST_DATA only: root alternatives, known/unknown domains and no binary edits."""
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from integrations.video_masks.boundary_precision import BoundaryPolicy,derive_trimap,prepare,execute


def test_same_root_alternatives_not_votes_and_core_unchanged():
    base=np.zeros((32,32),np.uint8);base[8:24,8:24]=255
    other=base.copy();other[10,10]=0
    union=base.copy();snapshot=base.copy()
    result=derive_trimap(union,base,[dict(mask=other,root_seed_decode_index=1),dict(mask=base,root_seed_decode_index=1)],0)
    assert result['stats']['admitted_roots']==2
    assert result['trimap'][16,16]==255 and result['trimap'][8,16]==128
    assert result['trimap'][3,3]==0
    assert np.array_equal(base,snapshot)
    assert result['stats']['source_status']=='MULTI_ROOT_AGREEMENT_NOT_TRUTH'


def test_disagreement_inside_union_is_unknown_never_fabricated_background():
    base=np.zeros((32,32),np.uint8);base[5:27,5:27]=255
    second=base.copy();second[14:18,14:18]=0
    result=derive_trimap(base,base,[dict(mask=second,root_seed_decode_index=1)],0)
    assert result['trimap'][15,15]==128 and result['validity'][15,15]==0
    assert set(np.unique(result['trimap']))<={0,128,255}
    assert np.array_equal(result['validity']!=0,result['trimap']!=128)


def test_single_source_and_low_iou_not_independent_support():
    base=np.zeros((32,32),np.uint8);base[8:24,8:24]=255
    bad=np.zeros_like(base);bad[1:3,1:3]=255
    union=base|bad
    result=derive_trimap(union,base,[dict(mask=bad,root_seed_decode_index=9)],0)
    assert result['stats']['admitted_roots']==1
    assert result['stats']['source_status']=='SINGLE_SOURCE_NOT_TRUTH'
    assert result['rejected_sources'][0]['reason']=='BELOW_SOURCE_IOU'
    assert result['trimap'][1,1]==128


@pytest.mark.parametrize('value',[True,float('nan'),-1])
def test_invalid_margin(value):
    with pytest.raises(ValueError):BoundaryPolicy(margin_pixels=value)


def fixture_source(tmp_path):
    source=tmp_path/'source';(source/'frames').mkdir(parents=True)
    mask=source/'mask.png';a=np.zeros((24,24),np.uint8);a[5:19,5:19]=255;Image.fromarray(a).save(mask)
    Image.new('RGB',(24,24)).save(source/'frames/000000.jpg')
    import hashlib
    digest=hashlib.sha256(mask.read_bytes()).hexdigest()
    for name,data in [('RESULT.json',dict(mode='union',status='FULL_TIMELINE_FUSED_EXPORTED')),
                      ('ASSEMBLY.json',dict(rows=[dict(decode_index=0,mask_path=str(mask),mask_sha256=digest,selected_root_seed_decode_index=0)])),
                      ('FUSION.json',dict(rows=[dict(decode_index=0,mode='union',base_mask_path=str(mask),base_mask_sha256=digest,support_evidence=[])]))]:
        (source/name).write_text(json.dumps(data),encoding='utf8')
    return source


def test_frozen_plan_drift_refuses_execution(tmp_path):
    source=fixture_source(tmp_path);out=tmp_path/'review';prepare(source,out,[0])
    (source/'mask.png').write_bytes(b'drift')
    with pytest.raises(ValueError,match='drift'):execute(out)
    assert not (out/'START.json').exists()


def test_real_cpu_fixture_exports_only_diagnostic_domain_and_once(tmp_path):
    source=fixture_source(tmp_path);out=tmp_path/'review';prepare(source,out,[0]);before=(source/'mask.png').read_bytes()
    result=execute(out)
    assert result['published_for_reconstruction'] is False
    assert result['new_model_executions']==result['new_gpu_executions']==0
    assert result['summary']['union_diff_pixels']==0
    assert (out/'union_upper_bound/000000.png').read_bytes()==before
    assert (source/'mask.png').read_bytes()==before
    with pytest.raises(ValueError,match='already attempted'):execute(out)
