"""Pixel-scale target competition; ambiguity ends identities, never merges them."""
import numpy as np


def default_policy():
    return dict(schema='object-track-identity-v1', target_separation_px=1.0,
                max_active=600, conflict_action='end_all_conflicting_ids',
                uncertainty='NOT_RESOLVED', merge_ids=False)


def validate_policy(policy):
    if (type(policy) is not dict or policy != default_policy()
            or type(policy.get('target_separation_px')) is not float
            or type(policy.get('max_active')) is not int
            or type(policy.get('merge_ids')) is not bool):
        raise ValueError('unsupported object track identity policy')


def conflict_pairs(points, policy):
    """Index pairs at <=1 pixel; proximity means ambiguous, not equal identity."""
    validate_policy(policy)
    q = np.asarray(points, dtype=float)
    if q.size == 0:
        return []
    if q.ndim != 2 or q.shape[1] != 2 or len(q) > policy['max_active'] or not np.isfinite(q).all():
        raise ValueError('bounded finite Nx2 target positions required')
    delta = q[:, None, :] - q[None, :, :]
    near = np.sum(delta*delta, axis=2) <= policy['target_separation_px']**2
    return np.argwhere(np.triu(near, 1)).tolist()


def isolate_targets(points, reasons, policy):
    """Only compete successful proposals; neither scores nor ID order picks a winner."""
    validate_policy(policy)
    q = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(q) != len(reasons) or len(q) > policy['max_active']:
        raise ValueError('target/reason count differs or budget exceeded')
    result = list(reasons)
    ids = [i for i, reason in enumerate(result) if reason == 'accepted']
    for a, b in conflict_pairs(q[ids], policy):
        result[ids[a]] = result[ids[b]] = 'target_identity_ambiguous'
    return result


def require_unique_targets(points, policy):
    pairs = conflict_pairs(points, policy)
    if pairs:
        raise ValueError('unresolved target identity conflict before native geometry')
