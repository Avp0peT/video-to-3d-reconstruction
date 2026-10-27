"""Frozen training/holdout diagnostics, without correspondence truth or quality gates."""
import numpy as np


def _pixels(raw):
    try:
        for row in raw:
            if len(row) != 4 or any(isinstance(v, (bool, np.bool_)) or
                    not isinstance(v, (int, float, np.integer, np.floating)) for v in row):
                raise ValueError('finite real Nx4 pixels required; bool rejected')
        a = np.array(raw, dtype=float, copy=True)
    except (TypeError, OverflowError) as exc:
        raise ValueError('finite real Nx4 pixels required') from exc
    if (a.ndim != 2 or a.shape[1] != 4 or not 12 <= len(a) <= 4096
            or not np.isfinite(a).all() or np.max(np.abs(a)) > 1e9):
        raise ValueError('12..4096 finite Nx4 pixel observations within 1e9 required')
    return a


def _split(n, train, holdout):
    for indices in (train, holdout):
        if not isinstance(indices, (list, tuple)) or any(type(i) is not int for i in indices):
            raise ValueError('frozen integer index lists required')
        if len(set(indices)) != len(indices) or any(i < 0 or i >= n for i in indices):
            raise ValueError('unique in-range indices required')
    if len(train) < 8 or len(holdout) < 2 or set(train) & set(holdout) or set(train) | set(holdout) != set(range(n)):
        raise ValueError('train>=8 and holdout>=2 must be disjoint and cover all rows')


def fit_affine_flat(train):
    """Fit one rank-three affine hyperplane in pixel-valued 4D."""
    mean = np.mean(train, axis=0)
    _, singular, vt = np.linalg.svd(train-mean, full_matrices=False)
    if not np.isfinite(singular).all() or singular[0] <= 0 or singular[2] <= singular[0]*1e-6:
        raise ValueError('affine rank-three support is degenerate')
    normal = vt[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return dict(mean=mean.tolist(), normal=normal.tolist(), singular_values=singular.tolist())


def _normalize(points):
    mean = points.mean(axis=0)
    centered = points-mean
    distance = np.mean(np.linalg.norm(centered, axis=1))
    singular = np.linalg.svd(centered, compute_uv=False)
    if distance <= 0 or singular[1] <= singular[0]*1e-6:
        raise ValueError('image point support is near-collinear or coincident')
    scale = np.sqrt(2)/distance
    transform = np.array([[scale, 0, -scale*mean[0]], [0, scale, -scale*mean[1]], [0, 0, 1]])
    return centered*scale, transform


def fit_fundamental(train):
    """Hartley-normalized eight-point training with rank-two projection."""
    if len(train) < 8:
        raise ValueError('fundamental training requires at least eight rows')
    a, ta = _normalize(train[:, :2]); b, tb = _normalize(train[:, 2:])
    x, y = a.T; u, v = b.T
    design = np.column_stack((u*x, u*y, u, v*x, v*y, v, x, y, np.ones(len(a))))
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    if singular[7] <= singular[0]*1e-6:
        raise ValueError('fundamental design lacks rank-eight support')
    f = vt[-1].reshape(3, 3)
    left, values, right = np.linalg.svd(f)
    if values[1] <= values[0]*1e-6:
        raise ValueError('fundamental rank-two support is degenerate')
    values[2] = 0
    f = tb.T @ (left @ np.diag(values) @ right) @ ta
    norm = np.linalg.norm(f)
    if not np.isfinite(f).all() or norm <= 0:
        raise ValueError('nonfinite fundamental estimate')
    f /= norm
    if f.ravel()[np.argmax(np.abs(f))] < 0:
        f = -f
    return dict(F=f.tolist(), design_singular_values=singular.tolist())


def _residual(kind, model, points):
    if kind == 'affine_flat':
        return np.abs((points-np.array(model['mean'])) @ np.array(model['normal']))
    f = np.array(model['F'])
    a = np.column_stack((points[:, :2], np.ones(len(points))))
    b = np.column_stack((points[:, 2:], np.ones(len(points))))
    fa = a @ f.T; ftb = b @ f
    denominator = np.sum(fa[:, :2]**2 + ftb[:, :2]**2, axis=1)
    if np.any(denominator <= np.finfo(float).tiny):
        raise ValueError('undefined Sampson denominator')
    result = np.abs(np.sum(b*fa, axis=1))/np.sqrt(denominator)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite residual')
    return result


def _metrics(values):
    return dict(residual_pixels=values.tolist(), rms=float(np.sqrt(np.mean(values**2))),
                median=float(np.median(values)), max=float(values.max()))


def analyze(raw, train_indices, holdout_indices):
    """Validate a frozen split; train each model once and evaluate fixed negative controls.

    Right-view holdout rows are cyclically shifted by one or reversed in the
    supplied holdout ordering. No control fits or updates either training model.
    Returned model parameters and index permutations are inspectable.
    """
    pixels = _pixels(raw); _split(len(pixels), train_indices, holdout_indices)
    train = pixels[list(train_indices)]; holdout = pixels[list(holdout_indices)]
    orders = dict(cyclic_one=np.roll(np.arange(len(holdout)), 1),
                  reverse=np.arange(len(holdout)-1, -1, -1))
    result = dict(train_indices=list(train_indices), holdout_indices=list(holdout_indices),
                  quality_status='NOT_EVALUATED', correspondence_truth='NOT_ESTABLISHED',
                  comparison_scope='Different model families and residual definitions; not matched degrees of freedom or exact-algorithm ranking')
    for kind, fit in (('affine_flat', fit_affine_flat), ('fundamental', fit_fundamental)):
        try:
            model = fit(train)
            correct = _residual(kind, model, holdout)
            controls = {}
            for name, order in orders.items():
                changed = holdout.copy(); changed[:, 2:] = holdout[order, 2:]
                residual = _residual(kind, model, changed); delta = residual-correct
                # Numerical equality tolerance is recorded, not a quality gate.
                tolerance = 1e-10*np.maximum(1., np.maximum(residual, correct))
                controls[name] = dict(**_metrics(residual), right_source_indices=[holdout_indices[i] for i in order],
                                     increase_count=int(np.sum(delta > tolerance)), decrease_count=int(np.sum(delta < -tolerance)),
                                     equal_count=int(np.sum(np.abs(delta) <= tolerance)), residual_difference_pixels=delta.tolist())
            result[kind] = dict(status='EVALUATED', model=model, correct=_metrics(correct), negative_controls=controls,
                                residual_definition='4D orthogonal pixel distance' if kind=='affine_flat' else 'sqrt Sampson squared distance in original pixels',
                                equality_tolerance='1e-10*max(1,correct,control)')
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            result[kind] = dict(status='NOT_EVALUATED', reason=str(exc), model=None)
    return result
