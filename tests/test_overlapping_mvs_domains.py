import numpy as np
from integrations.mvs_surface.overlapping_domains import world_groups, candidate_domains


def test_world_identity_order_stable():
    ids = np.array([9, 2, 7, 4]); xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [1, 1, 0]])
    assert world_groups(ids, xyz) == world_groups(ids[::-1], xyz[::-1])


def test_no_added_pixels_empty_unknown_and_no_cross_mask_jump():
    rgb = np.zeros((32, 40, 3), np.uint8); mask = np.ones((32, 40), bool); valid = mask.copy(); mask[:, 16:20] = False
    ds, counts = candidate_domains(rgb, mask, valid, [[[4, 4]], [], [[28, 4]]])
    assert not ds[1].any() and counts[1] == 0
    assert not ds[0][:, 20:].any() and not ds[2][:, :16].any()
    assert all(not (d & ~mask).any() for d in ds)


def test_overlap_with_nearby_seeds_and_invalid_cell_block():
    rgb = np.zeros((32, 32, 3), np.uint8); mask = np.ones((32, 32), bool); valid = mask.copy(); valid[0, 0] = False
    ds, _ = candidate_domains(rgb, mask, valid, [[[8, 8]], [[12, 8]], []])
    assert (ds[0] & ds[1]).any()
    assert all(not d[:4, :4].any() for d in ds)
