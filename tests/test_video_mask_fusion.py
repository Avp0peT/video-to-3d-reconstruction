import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from integrations.demo_pipeline.pipeline import sha
from integrations.video_masks.fusion import combine_rows

def fixture(tmp_path):
    base=np.zeros((20,20),np.uint8);base[3:17,3:17]=255;base[9:11,9:11]=0
    support=base.copy();support[9:11,9:11]=255;support[1,1]=255
    a,b=tmp_path/'base.png',tmp_path/'support.png'
    Image.fromarray(base).save(a);Image.fromarray(support).save(b)
    row=dict(decode_index=0,status='ORIGINAL_RETAINED',original_status='OK',mask_path=str(a),
        mask_sha256=sha(a),selected_root_seed_decode_index=0,warning_flags=[])
    item=dict(mask_path=str(b),mask_sha256=sha(b),root_seed_decode_index=17,direction='reverse')
    run=tmp_path/'run';run.mkdir()
    return run,dict(width=20,height=20),row,item,base,support

def test_union_provenance_adds_inside_and_outside_without_changing_original(tmp_path):
    run,plan,row,item,base,support=fixture(tmp_path)
    before=Path(row['mask_path']).read_bytes()
    result,summary=combine_rows(run,plan,[row],{0:[item]},'union')
    assert summary['changed_frames']==1 and summary['added_pixels']==5
    assert summary['removed_pixels']==0 and summary['holes_after']==0
    assert result[0]['status']=='FUSED_UNION_CANDIDATE'
    assert result[0]['original_status']=='OK' and result[0]['pre_fusion_status']=='ORIGINAL_RETAINED'
    assert Path(row['mask_path']).read_bytes()==before
    assert np.array_equal(np.asarray(Image.open(result[0]['mask_path'])),support)
    assert result[0]['fusion']['support_evidence'][0]['root_seed_decode_index']==17
    assert row['warning_flags']==[] and 'fusion' not in row
    assert json.loads((run/'FUSION.json').read_text())['summary']==summary

def test_missing_second_method_keeps_mask_and_status(tmp_path):
    run,plan,row,item,base,support=fixture(tmp_path)
    row['status']='SINGLE_DIRECTION_CANDIDATE';row['original_status']='NO_MATCH'
    result,summary=combine_rows(run,plan,[row],{},'union')
    assert summary['added_pixels']==0 and result[0]['mask_path']==row['mask_path']
    assert result[0]['original_status']=='NO_MATCH'
    assert not result[0]['fusion']['has_original_image_segmentation']

def test_drift_and_nonbinary_fail_closed(tmp_path):
    run,plan,row,item,base,support=fixture(tmp_path)
    Path(item['mask_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash drift'):
        combine_rows(run,plan,[row],{0:[item]})

def test_public_fusion_dispatch(tmp_path):
    from scripts.run_photo_demo import main
    with patch('integrations.video_masks.fusion.execute',return_value=dict(status='FULL_TIMELINE_FUSED_EXPORTED',
               automatic_boundary_review=dict(required=True,status='AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED'))) as call:
        assert main(['fuse-masks','--run',str(tmp_path/'s'),'--output',str(tmp_path/'o')])==0
        assert call.call_args.args==(tmp_path/'s',tmp_path/'o','union')

def test_full_video_can_disable_automatic_fusion(tmp_path):
    from scripts.propagate_video_masks import main
    with patch('integrations.video_masks.full_pipeline.execute',return_value=dict(status='PREPARED')) as call:
        assert main(['--pipeline',str(tmp_path/'p'),'--output',str(tmp_path/'o'),'--full-video','--mask-fusion','none'])==0
        assert call.call_args.args[-1]=='none'

def test_fusion_color_is_not_bidirectional_agreement():
    from integrations.video_masks.video_export import overlay_annotation
    assert overlay_annotation(0,'FUSED_UNION_CANDIDATE',False,True)[1]==(190,130,255)

def test_adjacent_metrics_recomputed_from_fused_pixels(tmp_path):
    import copy
    run,plan,row,item,base,support=fixture(tmp_path)
    second=copy.deepcopy(row)
    second.update(decode_index=1,adjacent_area_ratio=99,adjacent_centroid_distance=42,
                  warning_flags=['area_change_gt_1_5x_review_only','centroid_change_gt_0_15_review_only'])
    result,summary=combine_rows(run,plan,[row,second],{1:[item]})
    assert result[1]['pre_fusion_motion']==dict(adjacent_area_ratio=99,adjacent_centroid_distance=42)
    assert result[1]['adjacent_area_ratio']==pytest.approx(np.count_nonzero(support)/np.count_nonzero(base))
    distance=sum((a-b)**2 for a,b in zip(result[0]['centroid_normalized'],result[1]['centroid_normalized']))**.5
    assert result[1]['adjacent_centroid_distance']==pytest.approx(distance)
    assert result[1]['warning_flags']==[]
