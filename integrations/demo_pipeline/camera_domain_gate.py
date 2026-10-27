"""Necessary native inverse-domain checks; never a calibration accuracy claim.

Sampling the image domain cannot prove continuous/global invertibility. All stored
observations are checked, including untriangulated rows. No database is opened.
"""
from pathlib import Path
import hashlib
import numpy as np

GRID_SIDE = 9
ROUNDTRIP_TOLERANCE_PIXELS = 1e-3
# Numerical inversion allowance only (subpixel); never an accuracy threshold.
MAX_OBSERVATIONS = 2_000_000
MAX_IMAGES = 192
MAX_SOURCE_BYTES = 2 * 1024**3


def _snapshot(root):
    entries = {}
    total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('source symlinks refused')
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if total > MAX_SOURCE_BYTES or len(entries) >= 4096:
            raise ValueError('source hash bound exceeded')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        entries[path.relative_to(root).as_posix()] = dict(bytes=size, sha256=digest.hexdigest())
    if not entries:
        raise ValueError('empty model source')
    return entries


def audit_reconstruction(rec):
    """Inspect native Camera transforms without mutating the Reconstruction."""
    result = dict(schema='camera-domain-gate-v1', status='CAMERA_DOMAIN_REJECTED',
                  calibration_accuracy='NOT_EVALUATED', calibration_accepted=False,
                  continuous_domain_invertibility='NOT_PROVEN_BY_FINITE_SAMPLING',
                  policy=dict(grid_side=GRID_SIDE, roundtrip_tolerance_pixels=ROUNDTRIP_TOLERANCE_PIXELS,
                              max_observations=MAX_OBSERVATIONS, max_images=MAX_IMAGES), cameras=[], errors=[])
    cameras = dict(rec.cameras)
    images = dict(rec.images)
    count = sum(len(image.points2D) for image in images.values())
    if not cameras or not images or len(images) > MAX_IMAGES or len(cameras) > MAX_IMAGES or count > MAX_OBSERVATIONS:
        result['errors'].append('empty camera/image set or image/camera/observation bound exceeded')
        return result
    if any(image.camera_id not in cameras for image in images.values()):
        result['errors'].append('image references missing camera')
        return result
    for cid, camera in sorted(cameras.items()):
        row = dict(camera_id=int(cid), model=str(camera.model_name), observation_count=0,
                   grid_count=0, invalid_count=0, failures=[], max_roundtrip_error_pixels=0.)
        result['cameras'].append(row)
        def fail(kind, where, reason):
            row['invalid_count'] += 1
            if len(row['failures']) < 32:
                row['failures'].append(dict(kind=kind, location=where, reason=reason))
        try:
            params = np.asarray(camera.params, dtype=float)
            focals = params[np.asarray(camera.focal_length_idxs(), dtype=int)]
            if not np.isfinite(params).all() or not len(focals) or not np.isfinite(focals).all() or (focals <= 0).any():
                raise ValueError('camera parameters/focal invalid')
            width, height = int(camera.width), int(camera.height)
            if width <= 0 or height <= 0:
                raise ValueError('invalid camera dimensions')
        except Exception as exc:
            fail('intrinsics', None, str(exc))
            continue
        def check(xy, kind, where):
            row[kind + '_count'] += 1
            try:
                pixel = np.asarray(xy, dtype=float)
                if pixel.shape != (2,) or not np.isfinite(pixel).all():
                    raise ValueError('invalid source pixels')
                inverse = camera.cam_from_img(pixel)
                ray = np.asarray(inverse, dtype=float)
                if ray.shape != (2,) or not np.isfinite(ray).all():
                    raise ValueError('nonfinite/absent native inverse ray')
                if str(camera.model_name) == 'SIMPLE_RADIAL':
                    # Jacobian eigenvalues along the center-to-ray segment. For
                    # one radial coefficient their minima occur at an endpoint.
                    k = float(params[3]); r2 = float(ray @ ray)
                    if min(1., 1. + k*r2, 1. + 3.*k*r2) <= 0.:
                        raise ValueError('radial map is nonmonotone/singular on inverse-ray segment')
                elif str(camera.model_name) not in ('SIMPLE_PINHOLE', 'PINHOLE'):
                    raise ValueError('camera model lacks supported monotonic-domain audit')
                projected = np.asarray(camera.img_from_cam(np.r_[ray, 1.]), dtype=float)
                if projected.shape != (2,) or not np.isfinite(projected).all():
                    raise ValueError('nonfinite/absent native forward pixels')
                error = float(np.linalg.norm(projected - pixel))
                row['max_roundtrip_error_pixels'] = max(row['max_roundtrip_error_pixels'], error)
                if error > ROUNDTRIP_TOLERANCE_PIXELS:
                    raise ValueError('native inverse/forward roundtrip discrepancy')
            except Exception as exc:
                fail(kind, where, str(exc))
        for iid, image in sorted(images.items()):
            if image.camera_id == cid:
                for index, point in enumerate(image.points2D):
                    check(point.xy, 'observation', dict(image_id=int(iid), point2D_idx=index))
        # Native image-domain boundary convention [0,width] x [0,height], includes corners.
        for y in np.linspace(0., float(height), GRID_SIDE):
            for x in np.linspace(0., float(width), GRID_SIDE):
                check([x, y], 'grid', [float(x), float(y)])
    if not result['errors'] and all(row['invalid_count'] == 0 for row in result['cameras']):
        result['status'] = 'CAMERA_DOMAIN_VALID'
    return result


def audit_camera_domain(model_path, *, pycolmap_module=None):
    """Hash the entire model directory around one read-only native load/audit."""
    source = Path(model_path)
    if not source.is_absolute() or source.is_symlink():
        raise ValueError('absolute nonsymlink model directory required')
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError('model directory required')
    before = _snapshot(source)
    try:
        if pycolmap_module is None:
            import pycolmap as pycolmap_module
        rec = pycolmap_module.Reconstruction(str(source))
        result = audit_reconstruction(rec)
        result['backend_version'] = str(getattr(pycolmap_module, '__version__', 'unknown'))
    except Exception as exc:
        result = dict(schema='camera-domain-gate-v1', status='CAMERA_DOMAIN_REJECTED',
                      calibration_accuracy='NOT_EVALUATED', calibration_accepted=False,
                      errors=[type(exc).__name__ + ': ' + str(exc)])
    try:
        after = _snapshot(source)
        unchanged = before == after
    except Exception as exc:
        after = None
        unchanged = False
        result.setdefault('errors', []).append('source rehash failed: ' + str(exc))
    result.update(source_model=str(source), source_bindings_before=before,
                  source_bindings_after=after, source_unchanged=unchanged,
                  native_model_writes=0, sqlite_opened=False)
    if not unchanged:
        result['status'] = 'SOURCE_DRIFT_REFUSED'
    return result
