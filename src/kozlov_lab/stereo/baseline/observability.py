"""Local saved-XYZ engineering diagnostics, with fixed cameras and no fitting.

World length units are unknown. The optional prediction response assumes unit,
independent noise in each training raw-pixel coordinate. It is a linear response,
not physical uncertainty, a confidence interval, or geometry acceptance.
"""
import json
import numpy as np

from . import multiview_fixed as fixed
from . import radial_diagnostics as radial


def analyze(xyz, observations, predict_observation=None):
    """Return JSON-safe parallax, Jacobian SVD and optional unit-noise response.

    Rank uses singular values strictly greater than s_max * 1e-10. Ray
    directions come from the original measured pixels, not the candidate XYZ.
    Rank deficiency retains diagnostics but refuses prediction propagation.
    """
    try:
        point = radial._array(xyz, (3,))
        obs = fixed._observations(observations)
        projections = [fixed.project(point, o) for o in obs]
        prediction = (None if predict_observation is None else
                      fixed.project(point, predict_observation))
        views = []
        centers, rays = [], []
        for o, projection in zip(obs, projections):
            ideal = radial.undistort_simple_radial(
                o['raw_pixels'][None, :], o['params'],
                max_ideal_radius=o['max_ideal_radius'])['ideal_pixels'][0]
            uv = (ideal-o['params'][1:3])/o['params'][0]
            ray = o['R'].T @ np.array([uv[0], uv[1], 1.])
            ray /= np.linalg.norm(ray)
            center = -o['R'].T @ o['t']
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(ray)):
                raise ValueError('nonfinite camera center or world ray')
            centers.append(center)
            rays.append(ray)
            views.append(dict(image_id=o['image_id'], camera_center_world=center.tolist(),
                              normalized_world_ray=ray.tolist(),
                              candidate_raw_pixels=projection['pixels'].tolist(),
                              candidate_camera_depth=projection['depth']))
        pairs = []
        for i in range(len(obs)):
            for j in range(i+1, len(obs)):
                pairs.append(dict(image_ids=[obs[i]['image_id'], obs[j]['image_id']],
                                  ray_separation_degrees=float(np.degrees(np.arccos(
                                      np.clip(np.dot(rays[i], rays[j]), -1., 1.)))),
                                  baseline_length=float(np.linalg.norm(centers[i]-centers[j]))))
        jacobian = np.vstack([p['jacobian'] for p in projections])
        u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        if singular[0] <= 0:
            raise ValueError('zero stacked projection Jacobian')
        rank = int(np.sum(singular > singular[0]*1e-10))
        response = None
        if prediction is not None and rank == 3:
            inverse = (vt.T/singular) @ u.T
            mapping = prediction['jacobian'] @ inverse
            covariance = mapping @ mapping.T
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            response = dict(label='linear_unit_pixel_noise_response',
                            training_noise='unit independent raw-pixel coordinate noise',
                            covariance=covariance.tolist(), covariance_unit='pixel^2',
                            sqrt_eigenvalue_axes=np.sqrt(np.maximum(eigenvalues, 0.)).tolist(),
                            axes_unit='pixels',
                            candidate_raw_pixels=prediction['pixels'].tolist())
        result = dict(status='RANK_DEFICIENT' if rank < 3 else 'DIAGNOSTIC_EVALUATED',
                      reason='stacked raw-pixel Jacobian rank below 3' if rank < 3 else None,
                      xyz=point.tolist(), observations=views, pairs=pairs,
                      world_length_unit='unknown world unit', ray_angle_unit='degrees',
                      jacobian_unit='pixels/world unit', singular_values=singular.tolist(),
                      singular_value_ratio=float(singular[-1]/singular[0]), rank=rank,
                      rank_relative_threshold=1e-10, weak_world_direction=vt[-1].tolist(),
                      prediction_response=response,
                      scope='fixed-camera local engineering diagnostic; no fitting or geometry acceptance')
        # Reject overflow anywhere, including derived lengths and covariance.
        json.dumps(result, allow_nan=False)
        return result
    except (ValueError, ArithmeticError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        return dict(status='DIAGNOSTIC_REFUSED', reason=str(exc), prediction_response=None)
