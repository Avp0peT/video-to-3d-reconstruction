import numpy as np
import pytest

from kozlov_lab.stereo.temporal_support import assess_temporal_support, boundary_distance

cv = pytest.importorskip("cv2")


def scene():
    image = np.random.default_rng(72).integers(0, 256, (128, 128), dtype=np.uint8)
    moved = cv.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (128, 128))
    points = np.array([[64, 64], [44, 44], [84, 44], [44, 84], [84, 84]], dtype=float)
    return image, moved, points, points + [2, 1]


def assess(a, b, p, q, **kwargs):
    return assess_temporal_support(a, b, p, q, np.zeros(len(p)), np.ones(len(p)),
                                   kwargs.pop("anchors", np.ones(len(p), bool)), cv, **kwargs)


def test_translation_and_input_immutability():
    arrays = scene()
    copies = [x.copy() for x in arrays]
    rows = assess(*arrays)
    assert rows[0]["supported"]
    assert rows[0]["neighbor_count"] == 4
    assert rows[0]["patch_ncc"] > .99
    assert rows[0]["local_motion_residual_px"] == 0
    assert rows[0]["neighbor_indices"] == [1, 2, 3, 4]
    assert rows[0]["agreeing_neighbor_indices"] == [1, 2, 3, 4]
    for i, row in enumerate(rows):
        assert i not in row["neighbor_indices"]
        assert i not in row["agreeing_neighbor_indices"]
    for value, old in zip(arrays, copies):
        np.testing.assert_array_equal(value, old)


@pytest.mark.parametrize("mode", ["blank", "occluded", "different"])
def test_appearance_failures(mode):
    a, b, p, q = scene()
    if mode == "blank":
        a[:] = 0
        b[:] = 0
    elif mode == "occluded":
        b[60:72, 60:72] = 0
    else:
        b = np.random.default_rng(89).integers(0, 256, b.shape, dtype=np.uint8)
    assert not assess(a, b, p, q)[0]["supported"]


def test_anchor_mask_and_no_self_support():
    arrays = scene()
    row = assess(*arrays, anchors=np.array([True, False, False, False, False]))[0]
    assert row["neighbor_count"] == 0
    assert "insufficient_neighbors" in row["reasons"]
    assert row["neighbor_indices"] == []
    assert row["agreeing_neighbor_indices"] == []


def test_neighbor_indices_exclude_ineligible_input_rows():
    a, b, p, q = scene()
    anchors = np.array([True, True, False, True, True])
    rows = assess(a, b, p, q, anchors=anchors)
    for i, row in enumerate(rows):
        for key in ("neighbor_indices", "agreeing_neighbor_indices"):
            assert i not in row[key]
            assert 2 not in row[key]
            assert all(anchors[index] for index in row[key])
    assert rows[0]["neighbor_indices"] == [1, 3, 4]


def test_collinear_anchors():
    a, b, _, _ = scene()
    p = np.array([[64, 64], [34, 44], [49, 44], [79, 44], [94, 44]], float)
    row = assess(a, b, p, p + [2, 1])[0]
    assert not row["supported"]
    assert "collinear_neighbors" in row["reasons"]


def test_motion_disagreement_with_measured_matching_patches():
    a, b, p, q = scene()
    # Give the candidate an actual matching endpoint with a different motion.
    q[0] += [9, 0]
    b[61:70, 71:80] = a[60:69, 60:69]
    row = assess(a, b, p, q)[0]
    assert row["patch_ncc"] > .99
    assert row["local_motion_residual_px"] == 9
    assert "local_motion_disagreement" in row["reasons"]


@pytest.mark.parametrize("mode", ["nan", "edge", "status"])
def test_invalid_candidates(mode):
    a, b, p, q = scene()
    kwargs = {}
    if mode == "nan":
        q[0, 0] = np.nan
    elif mode == "edge":
        p[0] = [2, 2]
    else:
        kwargs["flow_valid"] = [False, True, True, True, True]
    assert not assess(a, b, p, q, **kwargs)[0]["supported"]


def test_failed_anchors_cannot_support_candidate():
    a, b, p, q = scene()
    rows = assess_temporal_support(a, b, p, q, [0, 2, 0, 0, 0], np.ones(5), np.ones(5, bool), cv)
    assert not rows[0]["supported"]
    assert rows[0]["neighbor_count"] == 3


def test_boundary_distance_and_empty_mask():
    mask = np.zeros((21, 21), np.uint8)
    mask[10, 10] = 255
    old = mask.copy()
    distance = boundary_distance(mask, cv)
    assert distance[10, 10] == 0
    assert distance[10, 16] == 6
    assert distance[10, 17] > 6
    np.testing.assert_array_equal(mask, old)
    assert np.isinf(boundary_distance(np.zeros_like(mask), cv)).all()


def test_cannot_relax_profile():
    with pytest.raises(ValueError, match="cannot relax"):
        assess(*scene(), profile={"patch_ncc_min": .1})


def test_subpixel_centers_are_actually_sampled():
    a, _, p, _ = scene()
    p += [.25, .375]
    centers = []

    class RecordingCV:
        @staticmethod
        def getRectSubPix(image, size, center):
            centers.append(center)
            return cv.getRectSubPix(image, size, center)

    rows = assess_temporal_support(a, a, p, p, np.zeros(5), np.ones(5),
                                   np.ones(5, bool), RecordingCV())
    assert rows[0]["supported"]
    assert centers[0] == (64.25, 64.375)


@pytest.mark.parametrize("fb,err", [(1.01, 1), (0, 25.01), (0, np.nan)])
def test_error_limits(fb, err):
    a, b, p, q = scene()
    rows = assess_temporal_support(a, b, p, q, np.full(5, fb),
                                   np.full(5, err), np.ones(5, bool), cv)
    assert not any(row["supported"] for row in rows)
