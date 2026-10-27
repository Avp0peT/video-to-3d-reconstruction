import copy
import unittest
from collections import Counter
from integrations.demo_pipeline.partial_surface import build_faces


def points(coords, views=3):
    return [dict(point_id=i, xyz=[x, y, 0.], observations=[
        dict(image_id=v, raw_pixels=[x * 10 + v, y * 10])
        for v in range(views)]) for i, (x,y) in enumerate(coords)]


class PartialSurfaceTests(unittest.TestCase):
    def test_three_view_grid_and_no_mutation(self):
        p = points([(x,y) for y in range(3) for x in range(3)])
        before = copy.deepcopy(p)
        faces, support, diag = build_faces(p)
        self.assertEqual(p, before)
        self.assertEqual(len(faces), 8)
        self.assertTrue(all(len(s) == 3 for s in support))
        self.assertEqual(len({tuple(sorted(f)) for f in faces}), len(faces))
        counts = Counter(tuple(sorted((f[a], f[b]))) for f in faces for a,b in ((0,1),(1,2),(2,0)))
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(diag['boundary_edge_count'], 8)
        directed = {}
        for f in faces:
            for a,b in ((0,1),(1,2),(2,0)):
                edge = (f[a], f[b])
                key = tuple(sorted(edge))
                if key in directed:
                    self.assertEqual(edge, directed[key][::-1])
                directed[key] = edge

    def test_reflected_anchor_has_consistent_winding(self):
        p = points([(x,y) for y in range(3) for x in range(3)])
        for point in p:
            for observation in point['observations']:
                observation['raw_pixels'][0] *= -1
        faces, _, _ = build_faces(p)
        self.assertEqual(len(faces), 8)
        import numpy as np
        for face in faces:
            q = np.array([p[i]['observations'][0]['raw_pixels'] for i in face])
            self.assertGreater(np.linalg.det(np.stack((q[1]-q[0], q[2]-q[0]))), 0)

    def test_duplicate_image_and_bool_identity_refused(self):
        p = points([(0,0),(1,0),(0,1)])
        p[0]['observations'].append(copy.deepcopy(p[0]['observations'][0]))
        with self.assertRaises(ValueError):
            build_faces(p)
        for identity in ('point_id', 'image_id'):
            p = points([(0,0),(1,0),(0,1)])
            if identity == 'point_id':
                p[0]['point_id'] = True
            else:
                p[0]['observations'][0]['image_id'] = True
            with self.assertRaises(ValueError):
                build_faces(p)

    def test_distant_components_and_large_gap_not_bridged(self):
        coords = [(0,0),(1,0),(0,1),(1,1),(20,0),(21,0),(20,1),(21,1),(100,100)]
        faces, _, _ = build_faces(points(coords))
        self.assertEqual(len(faces), 4)
        self.assertTrue(all(set(f) <= set(range(4)) or set(f) <= set(range(4,8)) for f in faces))

    def test_two_common_views_refused(self):
        faces, _, diag = build_faces(points([(0,0),(1,0),(0,1)], 2))
        self.assertEqual(faces, [])
        self.assertEqual(diag['refused_reasons']['fewer_than_three_common_views'], 1)

    def test_large_interior_hole_remains_open(self):
        coords = [(x,y) for y in range(11) for x in range(11)
                  if x in (0,1,9,10) or y in (0,1,9,10)]
        p = points(coords)
        faces, _, diag = build_faces(p)
        self.assertTrue(faces)
        # A triangle containing the centre would close the missing interior.
        import numpy as np
        for face in faces:
            q = np.array([coords[i] for i in face], dtype=float)
            matrix = np.vstack((q.T, np.ones(3)))
            weights = np.linalg.solve(matrix, [5.,5.,1.])
            self.assertFalse(np.all(weights >= -1e-9))
        self.assertGreater(diag['boundary_edge_count'], 0)

    def test_all_common_views_are_checked(self):
        p = points([(0,0),(1,0),(0,1)])
        p[2]['observations'][2]['raw_pixels'] = [1000,1000]
        self.assertEqual(build_faces(p)[0], [])

    def test_nonfinite_refused(self):
        for field in ('xyz', 'pixels'):
            p = points([(0,0),(1,0),(0,1)])
            if field == 'xyz':
                p[0]['xyz'][0] = float('nan')
            else:
                p[0]['observations'][0]['raw_pixels'][0] = float('inf')
            with self.assertRaises(ValueError):
                build_faces(p)

    def test_missing_anchor_points_remain_in_input(self):
        p = points([(0,0),(1,0),(0,1),(1,1)])
        p[3]['observations'] = []
        before = copy.deepcopy(p)
        faces, _, diag = build_faces(p)
        self.assertEqual(p, before)
        self.assertEqual(diag['input_point_count'], 4)
        self.assertTrue(all(3 not in f for f in faces))


if __name__ == '__main__':
    unittest.main()
