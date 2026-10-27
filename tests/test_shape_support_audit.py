import numpy as np
import pytest
from integrations.demo_pipeline import shape_support_audit as audit


class FakeCV:
    DIST_L2 = 2
    DIST_MASK_PRECISE = 0

    @staticmethod
    def distanceTransform(image, *_):
        yy, xx = np.indices(image.shape)
        anchors = np.argwhere(image == 0)
        return np.sqrt(np.min([(yy-y)**2 + (xx-x)**2 for y, x in anchors], axis=0))


def camera():
    return dict(width=5, height=5, K=np.eye(3), R=np.eye(3), C=np.zeros(3))


def policy(**changes):
    return dict(audit.POLICY, **changes)


def test_local_thin_extension_missing_despite_high_global_recall():
    mask = np.zeros((120, 80), bool)
    mask[:, :70] = True
    mask[50:60, 70:] = True
    raw = mask.copy(); raw[50:60, 70:] = False
    assert (raw & mask).sum()/mask.sum() > .98
    rows = audit.region_metrics(mask, np.ones_like(mask), mask, mask, raw, raw,
        policy=policy(grid_columns=8, grid_rows=12))
    warnings = [r for r in rows if 'BASELINE_SILHOUETTE_GAP' in r['reasons']]
    assert warnings and any(r['bbox'][0] >= 70 for r in warnings)


def test_background_removal_is_not_foreground_regression():
    mask = np.zeros((20, 20), bool); mask[5:15, 5:15] = True
    raw = np.ones_like(mask); final = mask.copy()
    rows = audit.region_metrics(mask, np.ones_like(mask), mask, mask, raw, final,
        policy=policy(grid_columns=1, grid_rows=1))
    assert rows[0]['regression_pixels'] == 0
    assert 'POSTPROCESS_FOREGROUND_REGRESSION' not in rows[0]['reasons']


def test_actual_foreground_loss_has_separate_regression_reason():
    mask = np.ones((20,20), bool); final = mask.copy(); final[:5] = False
    row = audit.region_metrics(mask, mask, mask, mask, mask, final,
        policy=policy(grid_columns=1, grid_rows=1))[0]
    assert row['regression_pixels'] == 100
    assert 'POSTPROCESS_FOREGROUND_REGRESSION' in row['reasons']


def test_invalid_mask_domain_is_unknown_not_missing_support():
    mask = np.ones((20,20),bool); valid = mask.copy(); valid[:10] = False
    row = audit.region_metrics(mask, valid, np.zeros_like(mask), np.zeros_like(mask),
        mask, mask, policy=policy(grid_columns=1, grid_rows=1))[0]
    assert row['status'] == 'UNKNOWN'
    assert row['reasons'] == ['INSUFFICIENT_VALID_MASK_AREA']
    assert 'depth_coverage' not in row


def test_unavailable_depth_and_points_are_none_not_zero():
    mask = np.ones((20,20),bool)
    row = audit.region_metrics(mask, mask, raw=mask, final=mask,
        policy=policy(grid_columns=1, grid_rows=1))[0]
    assert row['depth_coverage'] is None and row['points_coverage'] is None
    assert 'DEPTH_UNAVAILABLE' in row['reasons']
    assert 'LOW_STORED_DEPTH_SUPPORT' not in row['reasons']


def test_point_z_buffer_keeps_nearest_not_matching_back_point():
    # Same pixel (1,1), front Z1 disagrees; the back Z2 happens to match depth.
    pts = np.array([[1,1,1], [2,2,2]])
    depth = np.full((5,5),2.)
    out = audit.point_support(pts, camera(), depth, FakeCV)
    assert out['projected'].sum() == 1 and out['projected'][1,1]
    assert not out['depth_agreeing'].any()
    assert not out['depth_agreeing_near'].any()


def test_point_order_cannot_change_nearest_depth_agreement():
    pts = np.array([[2,2,2], [1,1,1]])
    depth = np.ones((5,5))
    out = audit.point_support(pts, camera(), depth, FakeCV)
    assert out['depth_agreeing'].sum() == 1


def test_behind_camera_offscreen_and_rounding_boundaries_do_not_leak():
    pts = np.array([[0,0,-1], [-.51,0,1], [4.5,0,1], [-.49,0,1], [4.49,4.49,1]])
    out = audit.point_support(pts, camera(), None, FakeCV)
    assert out['positive_camera_points'] == 4 and out['inside_camera_points'] == 2
    assert out['projected'].sum() == 2
    assert out['projected'][0,0] and out['projected'][4,4]
    assert out['depth_agreeing'] is None and out['depth_agreeing_near'] is None


def test_invalid_saved_depth_is_not_agreement():
    depth = np.ones((5,5));depth[1,1] = np.nan;depth[2,2] = 0
    out = audit.point_support([[1,1,1], [2,2,1]], camera(), depth, FakeCV)
    assert not out['depth_agreeing'].any()


def test_empty_points_never_fill_near_domain():
    out = audit.point_support(np.empty((0,3)), camera(), np.ones((5,5)), FakeCV)
    assert not out['projected_near'].any() and not out['depth_agreeing_near'].any()


@pytest.mark.parametrize('views', [[], [{'status':'UNKNOWN','regions':[]}],
    [{'status':'EVALUATED','regions':[]}],
    [{'status':'EVALUATED','regions':[{'status':'UNKNOWN','reasons':[]}]}]])
def test_unknown_evidence_never_promotes(views):
    result = audit.verdict(views, True)
    assert not result['candidate_promotable'] and result['status'] == 'REVIEW_REQUIRED'


def test_no_flags_is_still_not_quality_acceptance_and_candidate_required():
    views=[dict(status='EVALUATED',regions=[dict(status='MEASURED',reasons=[])])]
    assert audit.verdict(views,True)['status']=='NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE'
    assert not audit.verdict(views,False)['candidate_promotable']


def test_frame_edge_contact_is_risk_not_silent_complete():
    views=[dict(status='EVALUATED',mask_touches_frame=True,regions=[dict(status='MEASURED',reasons=[])])]
    result=audit.verdict(views,True)
    assert not result['candidate_promotable']
    assert 'SOURCE_FRAME_CLIPPING_RISK' in result['reasons']
