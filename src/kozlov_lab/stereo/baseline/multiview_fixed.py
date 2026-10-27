"""Fixed-camera DLT and bounded XYZ-only refinement in original radial pixels.

Standard multiview least squares, not Kozlov and not bundle adjustment. Inputs
are training observations only; intrinsics and camera poses never change.
"""
import numpy as np

from . import radial_diagnostics as radial


def _observation(observation):
    if not isinstance(observation, dict):
        raise ValueError('observation dictionary required')
    if type(observation.get('image_id')) is not int:
        raise ValueError('integer image_id required; bool rejected')
    return dict(image_id=observation['image_id'],
                raw_pixels=radial._array(observation['raw_pixels'], (2,)),
                params=radial._camera(observation['params'], 2),
                R=radial._rotation(observation['R']),
                t=radial._array(observation['t'], (3,)),
                max_ideal_radius=radial._domain(
                    radial._camera(observation['params'], 2)[3],
                    observation['max_ideal_radius']))


def _observations(observations, minimum=2):
    if not isinstance(observations, (list, tuple)) or not minimum <= len(observations) <= 4096:
        raise ValueError(f'{minimum}..4096 complete observations required')
    result = [_observation(o) for o in observations]
    ids = [o['image_id'] for o in result]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate image_id')
    return result


def project(xyz, observation):
    """Project a world XYZ; raises on invalid input/domain/cheirality."""
    point = radial._array(xyz, (3,))
    o = _observation(observation)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        camera = o['R'] @ point + o['t']
        if not np.all(np.isfinite(camera)) or camera[2] <= 0:
            raise ValueError('nonfinite or nonpositive camera depth')
        u, v = camera[:2] / camera[2]
        f, cx, cy, k = o['params']
        ideal = np.array([[f*u+cx, f*v+cy]])
        pixels = radial.distort_simple_radial(
            ideal, o['params'], max_ideal_radius=o['max_ideal_radius'])[0]
        factor = 1 + k*(u*u+v*v)
        distortion = f*np.array([[factor+2*k*u*u, 2*k*u*v],
                                 [2*k*u*v, factor+2*k*v*v]])
        pinhole = np.array([[1, 0, -u], [0, 1, -v]]) / camera[2]
        jacobian = distortion @ pinhole @ o['R']
    if not np.all(np.isfinite(jacobian)):
        raise ValueError('nonfinite projection Jacobian')
    return dict(pixels=pixels, depth=float(camera[2]), jacobian=jacobian)


def _system(xyz, observations):
    projections = [project(xyz, o) for o in observations]
    residuals = np.array([p['pixels']-o['raw_pixels']
                          for p, o in zip(projections, observations)])
    jacobian = np.vstack([p['jacobian'] for p in projections])
    cost = float(np.sum(residuals*residuals))
    if not np.isfinite(cost):
        raise ValueError('nonfinite residual cost')
    return residuals, jacobian, cost


def evaluate(xyz, observations):
    """Complete original-pixel residual diagnostic, or explicit refusal."""
    try:
        obs = _observations(observations, minimum=1)
        residuals, _, cost = _system(xyz, obs)
        norms = np.linalg.norm(residuals, axis=1)
        return dict(status='EVALUATED', residual_norms=norms.tolist(),
                    residual_vectors=residuals.tolist(), rms=float(np.sqrt(cost/len(obs))),
                    max=float(np.max(norms)), failure=None)
    except (ValueError, ArithmeticError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        return dict(status='PROJECTION_REFUSED', residual_norms=[], residual_vectors=[],
                    rms=None, max=None, failure=str(exc))


def fit(observations, max_iterations=20):
    """DLT initializer then at most max_iterations LM steps, accepting descent only.

    Cost is sum of squared original-pixel vector residuals. Each LM step has
    at most 12 damping trials. Exhaustion retains the best finite feasible XYZ
    and reports converged=False; no validation observation is consulted.
    """
    result = dict(status='REFUSED', xyz=None, initial_xyz=None, initial_cost=None,
                  final_cost=None, iterations=0, converged=False, reason=None,
                  train_image_ids=[])
    try:
        if type(max_iterations) is not int or not 1 <= max_iterations <= 100:
            raise ValueError('integer iteration bound 1..100 required')
        obs = _observations(observations)
        result['train_image_ids'] = [o['image_id'] for o in obs]
        centers = np.array([-o['R'].T @ o['t'] for o in obs])
        if np.max(np.linalg.norm(centers-centers[0], axis=1)) <= 1e-12:
            raise ValueError('coincident camera centers')
        rows = []
        for o in obs:
            ideal = radial.undistort_simple_radial(
                o['raw_pixels'][None, :], o['params'],
                max_ideal_radius=o['max_ideal_radius'])['ideal_pixels'][0]
            u, v = (ideal-o['params'][1:3])/o['params'][0]
            p = np.column_stack((o['R'], o['t']))
            for row in (u*p[2]-p[0], v*p[2]-p[1]):
                norm = np.linalg.norm(row)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError('degenerate DLT row')
                rows.append(row/norm)
        _, singular, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
        if singular[2] <= singular[0]*1e-10:
            raise ValueError('rank-deficient multiview triangulation')
        homogeneous = vt[-1]
        if abs(homogeneous[3]) <= 1e-12:
            raise ValueError('triangulation at infinity')
        xyz = homogeneous[:3]/homogeneous[3]
        residuals, jacobian, cost = _system(xyz, obs)
        if np.linalg.matrix_rank(jacobian, tol=np.linalg.norm(jacobian)*1e-10) < 3:
            raise ValueError('rank-deficient XYZ objective')
        result.update(initial_xyz=xyz.copy(), initial_cost=cost)
        damping = 1e-3
        reason = 'iteration bound reached'
        for iteration in range(max_iterations):
            gradient = jacobian.T @ residuals.ravel()
            hessian = jacobian.T @ jacobian
            if np.linalg.norm(gradient, ord=np.inf) <= 1e-8:
                result['converged'] = True
                reason = None
                break
            result['iterations'] = iteration+1
            accepted = False
            for _ in range(12):
                step = np.linalg.solve(hessian+damping*np.diag(
                    np.maximum(np.diag(hessian), 1e-12)), -gradient)
                try:
                    candidate = xyz+step
                    new_residuals, new_jacobian, new_cost = _system(candidate, obs)
                except (ValueError, ArithmeticError):
                    new_cost = np.inf
                if new_cost < cost:
                    old_cost = cost
                    xyz, residuals, jacobian, cost = candidate, new_residuals, new_jacobian, new_cost
                    damping = max(damping/3, 1e-12)
                    accepted = True
                    if np.linalg.norm(step) <= 1e-10*(1+np.linalg.norm(xyz)) or old_cost-cost <= 1e-12*max(1, old_cost):
                        result['converged'] = True
                        reason = None
                    break
                damping *= 10
            if result['converged']:
                break
            if not accepted:
                reason = 'bounded damping trials exhausted without descent'
                break
        result.update(status='FITTED', xyz=xyz.copy(), final_cost=cost, reason=reason)
    except (ValueError, ArithmeticError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        result.update(status='REFUSED', xyz=None, reason=str(exc))
    return result
