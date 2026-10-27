import numpy as np
import pytest

from integrations.video_masks.local_face_evidence import (
    LocalPolicy, closed_footprint_cells, evaluate_local_view)


def evaluate(triangle, *, mode='closed_cells', hole=None, gap_hole=None, expand_gap=False, policy=LocalPolicy()):
    mask = np.zeros((120,120),bool)
    mask[:,10:30] = True; mask[:,80:110] = True
    valid = np.ones_like(mask)
    gap = np.zeros_like(mask); gap[10:110,34:76] = True
    if expand_gap: gap[10:110,30:80] = True
    if hole is not None:
        valid[hole] = False; gap[hole] = False
    if gap_hole is not None: gap[gap_hole] = False
    v = np.column_stack((np.asarray(triangle,float),np.ones(3)))
    f = np.array([[0,1,2]])
    return evaluate_local_view(v,f,K=np.eye(3),R=np.eye(3),C=np.zeros(3),
        mask=mask,validity=valid,depth=None,gap_mask=gap,view_id='test',
        original_vertices=v,original_faces=f,parent_ids=np.array([0]),
        eligible=np.array([True]),policy=policy,footprint_mode=mode)


LONG = [[50.2,20.2],[54.8,20.2],[50.2,99.8]]


def test_long_triangle_passes_exact_cells_but_circle_rejects():
    old = evaluate(LONG,mode='disk'); new = evaluate(LONG)
    assert not old['negative'][0] and not old['whole_gap_disk'][0]
    assert new['negative'][0] and new['whole_gap_footprint'][0]
    assert new['actual_pixel_count'][0] == 4
    assert new['footprint_cells'][0] > 4
    assert not new['geometry_accepted']


@pytest.mark.parametrize('kwargs',[
    {'hole':(50,51)}, {'gap_hole':(50,51)},
])
def test_interior_unknown_or_crossed_gap_cell_blocks(kwargs):
    row = evaluate(LONG,**kwargs)
    assert not row['negative'][0]


def test_closed_contact_checks_cell_outside_triangle_centres():
    # Vertex at x=49.5 touches the closed cell centred on x=49.
    tri = [[49.5,30.2],[54.8,30.2],[49.5,80.2]]
    row = evaluate(tri,gap_hole=(40,49))
    assert not row['negative'][0] and not row['whole_gap_footprint'][0]


@pytest.mark.parametrize('tri',[
    [[30.1,40.1],[33.9,40.1],[30.1,43.9]],
    [[45.1,50.1],[65.1,50.1],[45.1,50.6]],
    [[50.1,50.1],[50.3,50.1],[50.1,50.3]],
    [[50,40],[51,41],[52,42]],
    [[-.5,40],[5,40],[5,45]],
    [[50,40],[90,40],[50,80]],
])
def test_boundary_few_centres_subpixel_degenerate_offscreen_crossing_fail(tri):
    assert not evaluate(tri)['negative'][0]


def test_clearance_remains_independent_of_gap_and_budget_fails_closed():
    near = evaluate([[30.6,40.2],[34.2,40.2],[34.2,60.2]],expand_gap=True)
    assert near['footprint_vertex_prefilter_rejected'][0]
    assert not near['footprint_evaluated'][0]
    assert not near['deep_background_footprint'][0] and not near['negative'][0]
    row = evaluate(LONG,policy=LocalPolicy(max_raster_pixels_per_face=10))
    assert not row['negative'][0] and not row['footprint_evaluated'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'PIXEL_BUDGET'


def test_closed_cells_order_orientation_contact_and_no_input_mutation():
    t = np.array([[5.5,5.5],[8.2,5.5],[5.5,8.2]])
    before=t.copy()
    a,error=closed_footprint_cells(t,20,20,100)
    b,error2=closed_footprint_cells(t[::-1],20,20,100)
    assert error is error2 is None
    assert np.array_equal(a,b) and np.array_equal(t,before)
    assert [5,5] in a.tolist()
    assert [8,8] not in a.tolist()  # Bbox is not the actual triangle footprint.


def test_helper_offscreen_degenerate_deadline_and_bad_mode():
    assert closed_footprint_cells(np.array([[-.5,5],[2,5],[2,8]]),20,20,100)[1] == 'OUT_OF_FRAME'
    assert closed_footprint_cells(np.array([[5,5],[6,6],[7,7]]),20,20,100)[1] == 'SUBPIXEL_OR_DEGENERATE'
    def stop(): raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError): closed_footprint_cells(np.array(LONG),120,120,10000,stop)
    with pytest.raises(ValueError): evaluate(LONG,mode='other')


def test_prefilter_skips_sat_only_for_a_necessarily_failing_vertex_cell(monkeypatch):
    from integrations.video_masks import local_face_evidence as module
    triangle = [[34.2,20.2],[54.8,20.2],[50.2,80.2]]
    cells,error = closed_footprint_cells(np.array(triangle),120,120,10000)
    assert error is None and [34,20] in cells.tolist()
    def forbidden(*args,**kwargs):
        raise AssertionError('necessarily failing vertex must skip SAT')
    monkeypatch.setattr(module,'closed_footprint_cells',forbidden)
    row = evaluate(triangle,gap_hole=(20,34))
    assert row['footprint_vertex_prefilter_rejected'][0]
    assert not row['footprint_evaluated'][0] and not row['negative'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'NO_WHOLE_GAP_PROOF'


def test_prefilter_does_not_approve_boundary_contact_or_interior_unknown():
    boundary = evaluate([[49.5,30.2],[54.8,30.2],[49.5,80.2]],gap_hole=(40,49))
    unknown = evaluate(LONG,hole=(50,51))
    for row in (boundary,unknown):
        assert not row['footprint_vertex_prefilter_rejected'][0]
        assert row['footprint_evaluated'][0] and not row['negative'][0]
    assert unknown['reason_names'][int(unknown['reason'][0])] == 'NO_KNOWN_DOMAIN_FOOTPRINT'


def test_prefilter_margin_failure_is_same_as_full_cells(monkeypatch):
    from integrations.video_masks import local_face_evidence as module
    triangle = [[30.6,40.2],[34.2,40.2],[34.2,60.2]]
    cells,error = closed_footprint_cells(np.array(triangle),120,120,10000)
    assert error is None and [31,40] in cells.tolist()
    def forbidden(*args,**kwargs): raise AssertionError('margin failure must skip SAT')
    monkeypatch.setattr(module,'closed_footprint_cells',forbidden)
    row = evaluate(triangle,expand_gap=True)
    assert row['footprint_vertex_prefilter_rejected'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'NO_DEEP_BACKGROUND_PROOF'
