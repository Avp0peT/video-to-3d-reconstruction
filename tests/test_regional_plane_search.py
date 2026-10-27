from copy import deepcopy
import cv2
import numpy as np
import pytest
from integrations.mvs_surface.regional_plane_search import propose_regional_depth, _fit


def scene():
    cv2.setNumThreads(2)
    h = w = 96
    k = np.array([[100., 0, 48], [0, 100., 48], [0, 0, 1.]])
    y, x = np.mgrid[:h, :w]
    rays = np.stack(((x-48)/100, (y-48)/100, np.ones((h,w))), axis=-1)
    normal = np.array([.25, -.15, 1.])
    def view(i, b):
        c = np.array([b, 0., 0.])
        z = ((2.-normal@c)/(rays@normal)).astype(np.float32)
        return dict(view_id=i, K=k.copy(), R=np.eye(3), C=c, image=np.full((h,w,3), 12, np.uint8), mask=np.ones((h,w), bool), depth=z)
    ref = view(0, 0.)
    truth = ref['depth'].copy()
    roi = np.zeros((h,w), bool)
    roi[40:56,40:56] = True
    ref['depth'][roi] = 0
    return ref, [view(i+1,b) for i,b in enumerate([.06,-.06,.12,-.12])], roi, truth


def run(ref, src, roi):
    return propose_regional_depth(ref, src, roi, [1., 4.])


def test_slanted_low_texture_translated_plane_recovered_without_mutation():
    ref, src, roi, truth = scene()
    before = deepcopy((ref, src))
    r = run(ref, src, roi)
    assert r['accepted'][roi].all()
    np.testing.assert_allclose(r['proposal_depth'][roi], truth[roi], rtol=1e-5)
    assert not r['proposal_depth'][~roi].any()
    assert set(r['source_ids']) == {1,2,3}
    assert r['heldout_source_ids'] == [4]
    assert r['heldout_validation']['per_source']['4']['support'] == roi.sum()
    assert r['per_scale_fits']['32'] and r['per_scale_fits']['96']
    assert r['policy']['window_sizes'] == [32,64,96]
    assert r['policy']['min_anchors'] == 12
    assert r['policy']['geometry_relative_tolerance'] == .02
    assert r['policy']['max_reprojection_error'] == 1.2
    assert not {'planes','min_ncc','patch_size'} & set(r['policy'])
    for a,b in zip([ref,*src], [before[0],*before[1]]):
        for key in ('depth','image','K','R','C','mask'):
            np.testing.assert_array_equal(a[key], b[key])


def test_heldout_changes_diagnostics_only():
    ref, src, roi, _ = scene()
    a = run(ref,src,roi)
    src[-1]['depth'][:] = 3.
    b = run(ref,src,roi)
    for key in ('accepted','proposal_depth','raw_best_depth','anchor_mask','chosen_scale'):
        np.testing.assert_array_equal(a[key],b[key])
    assert b['heldout_validation']['per_source']['4']['conflict'] > 0


@pytest.mark.parametrize('mode', ['missing','extrapolation','degenerate','gap','baddepth'])
def test_unsupported_cases_reject(mode):
    ref,src,roi,_ = scene()
    if mode == 'missing':
        ref['depth'][:] = 0
    elif mode == 'extrapolation':
        ref['depth'][:,38:] = 0
    elif mode == 'degenerate':
        ref['depth'][:47] = 0
        ref['depth'][48:] = 0
    elif mode == 'gap':
        ref['mask'][38:58,38:58] = False
        roi[:] = False
        roi[39:57,39:57] = ref['mask'][39:57,39:57]
    elif mode == 'baddepth':
        for s in src[:3]:
            s['depth'][:] = 3.
    r = run(ref,src,roi)
    assert not r['accepted'].any()


def test_component_gap_not_bridged():
    ref,src,roi,_ = scene()
    ref['mask'][:,47:49] = False
    roi &= ref['mask']
    r = run(ref,src,roi)
    assert not r['accepted'][:,46:50].any()


def test_fit_rejects_degenerate_and_nonplanar():
    x = np.arange(20.)
    assert _fit(x,x,np.ones(20)*2,10,10,32) is None
    y,x = np.mgrid[:8,:8]
    z = 2.+.3*((x+y)%2)
    assert _fit(x.ravel(),y.ravel(),z.ravel(),4,4,32) is None


def test_source_count_and_invalid_depth():
    ref,src,roi,_ = scene()
    with pytest.raises(ValueError):
        run(ref,src[:3],roi)
    ref['depth'][0,0] = np.nan
    with pytest.raises(ValueError):
        run(ref,src,roi)


def test_competing_hypotheses_fail_closed_even_with_geometric_support(monkeypatch):
    # Inject a second valid planar hypothesis to isolate arbitration from fitting.
    # Admission sources still support the original plane, so conflict arbitration
    # must independently prevent admission rather than average the hypotheses.
    from integrations.mvs_surface import regional_plane_search as module
    original = module._fit
    def competing(x,y,z,cx,cy,scale):
        fit = original(x,y,z,cx,cy,scale)
        if fit is not None and scale == 96:
            coef,hull,n,residual = fit
            return coef/1.05,hull,n,residual
        return fit
    monkeypatch.setattr(module,'_fit',competing)
    ref,src,roi,_ = scene()
    result = run(ref,src,roi)
    assert result['competing_planes'][roi].all()
    assert not result['accepted'].any()
    assert (result['reason'][roi] == 2).all()


@pytest.mark.parametrize('gap', ['hole','u'])
def test_rasterized_hull_cannot_cross_component_gap(gap):
    from integrations.mvs_surface.regional_plane_search import _hull_inside
    component = np.ones((32,32), bool)
    if gap == 'hole':
        component[13:19,13:19] = False
    else:
        component[:24,13:19] = False
    hull = np.array([[[3,3]],[[28,3]],[[28,28]],[[3,28]]],np.float32)
    assert not _hull_inside(hull,component,0,0)


def test_nearly_collinear_anchor_conditioning_rejected():
    x = np.linspace(0,31,40)
    y = x + .001*np.sin(x)
    assert _fit(x,y,np.ones(40)*2,16,16,32) is None


def test_known_background_veto_and_invalid_validity_unknown():
    ref,src,roi,_ = scene()
    # Third admission source is known background in the projected hole only.
    src[2]['mask'][38:58,30:60] = False
    r = run(ref,src,roi)
    assert r['background_veto'][roi].all()
    assert not r['accepted'].any()
    src[2]['validity'] = np.ones(roi.shape,bool)
    src[2]['validity'][38:58,30:60] = False
    r = run(ref,src,roi)
    assert not r['background_veto'][roi].any()
    assert r['accepted'][roi].all()


def test_heldout_no_depth_is_not_evaluable():
    ref,src,roi,_ = scene()
    src[-1]['depth'][:] = 0
    r = run(ref,src,roi)
    row = r['heldout_validation']['outcomes_per_source']['4']
    assert row['status'] == 'NOT_EVALUABLE'
    assert row['available_depth_pixels'] == row['support'] == 0
    assert row['unknown'] == int(roi.sum())
    assert r['accepted'][roi].all()


def test_observed_depth_discontinuity_splits_support_sets():
    from integrations.mvs_surface.regional_plane_search import _surface_sets
    anchors = np.ones((8,8),bool)
    depth = np.full((8,8),2.,np.float32)
    depth[:,4:] = 3.
    labels = _surface_sets(anchors,depth,lambda:None)
    assert len(np.unique(labels)) == 2
    assert labels[2,2] != labels[2,5]
