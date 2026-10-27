"""2D enclosing ellipse: floating Khachiyan solve, exact rational end certificate.

Kozlov 2025 p.149 defines an exact minimum-area ellipse centre and major-axis
length. This module supplies an area-bounded numerical approximation, NOT that
exact optimum. Algorithm: Todd & Yildirim, TYKhach.pdf, Algorithm 3.1. The final
containment/area certificate applies to the actual stored binary64 inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from fractions import Fraction
import math
from typing import Any

import numpy as np

MAX_POINTS = 4096
MAX_ITERATIONS = 100_000
MAX_WORK = 100_000_000
PROOF_SCOPE = "stored_binary64_inputs"


class EllipseInputError(ValueError):
    pass


def _points(value: Any) -> np.ndarray:
    # Reject bool/string coercion before NumPy converts mixed Python lists.
    def valid(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return all(valid(x) for x in v)
        if isinstance(v, (bool, np.bool_)):
            return False
        if isinstance(v, (int, np.integer)):
            # Must precede np.asarray: a mixed int/float list can already round
            # the integer while constructing the array.
            try:
                converted = float(v)
                return math.isfinite(converted) and int(converted) == int(v)
            except OverflowError:
                return False
        return isinstance(v, (float, np.floating))
    try:
        if not valid(value):
            raise EllipseInputError("coordinates must be real numbers, not booleans or strings")
        source = np.asarray(value)
        if source.ndim != 2 or source.shape[1] != 2 or not 1 <= len(source) <= MAX_POINTS:
            raise EllipseInputError("expected 1..4096 two-dimensional points")
        if source.dtype.kind not in "fiu" or source.dtype.itemsize > 8:
            raise EllipseInputError("input must have binary64-compatible numeric precision")
        p = np.array(source, dtype=np.float64, copy=True)
        if not np.isfinite(p).all():
            raise EllipseInputError("coordinates must be finite binary64 numbers")
        if source.dtype.kind in "iu" and any(int(x) != int(y) for x, y in zip(source.ravel(), p.ravel())):
            raise EllipseInputError("integer coordinates are not exactly representable in binary64")
        return p
    except (TypeError, OverflowError) as exc:
        raise EllipseInputError("invalid point array") from exc
    except ValueError as exc:
        if isinstance(exc, EllipseInputError):
            raise
        raise EllipseInputError("invalid/ragged point array") from exc


def _f(value) -> Fraction:
    return Fraction.from_float(float(value))


def _rational(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _exact_points(points: np.ndarray) -> list[tuple[Fraction, Fraction]]:
    return [(_f(x), _f(y)) for x, y in points]


def _upper_sqrt(value: Fraction) -> float:
    """Binary64 upper bound, checked against the rational square."""
    if value < 0:
        raise ArithmeticError("negative square")
    result = math.sqrt(float(value))
    if not math.isfinite(result):
        raise ArithmeticError("square root display is outside binary64 range")
    for _ in range(4):
        if _f(result)**2 >= value:
            return result
        result = float(np.nextafter(result, math.inf))
    raise ArithmeticError("could not round square root upward")


def _exact_q(points, center: np.ndarray, factor: np.ndarray) -> list[Fraction]:
    c = [_f(v) for v in center]
    b = [[_f(v) for v in row] for row in factor]
    result = []
    for x, y in points:
        dx, dy = x - c[0], y - c[1]
        result.append((b[0][0]*dx+b[0][1]*dy)**2 + (b[1][0]*dx+b[1][1]*dy)**2)
    return result


def _covariance_det(points, weights: np.ndarray) -> Fraction:
    # Compute FROM stored input coordinates and stored weights, never from a
    # rounded NumPy covariance. Rational normalization produces a true simplex.
    u = [_f(v) for v in weights]
    total = sum(u)
    if total <= 0 or any(v < 0 for v in u):
        raise ArithmeticError("invalid dual weights")
    u = [v/total for v in u]
    mx = sum(w*p[0] for w, p in zip(u, points))
    my = sum(w*p[1] for w, p in zip(u, points))
    xx = sum(w*(p[0]-mx)**2 for w, p in zip(u, points))
    yy = sum(w*(p[1]-my)**2 for w, p in zip(u, points))
    xy = sum(w*(p[0]-mx)*(p[1]-my) for w, p in zip(u, points))
    return xx*yy-xy*xy


@dataclass(frozen=True)
class EllipseResult:
    status: str
    center: np.ndarray | None
    factor: np.ndarray | None
    area: float | None
    semiaxes: np.ndarray | None
    major_axis_length: float | None
    iterations: int
    unique_points: np.ndarray
    weights: np.ndarray | None
    _certificate: dict

    def __post_init__(self):
        # Back with immutable bytes, not only a reversible writeable=False flag.
        for name in ("center", "factor", "semiaxes", "unique_points", "weights"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                sealed = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
                object.__setattr__(self, name, sealed)
        object.__setattr__(self, "_certificate", copy.deepcopy(self._certificate))

    @property
    def certificate(self) -> dict:
        return copy.deepcopy(self._certificate)

    @property
    def converged(self) -> bool:
        return self.status in ("APPROXIMATE_CERTIFIED", "DEGENERATE_POINT", "DEGENERATE_SEGMENT")

    def contains(self, points: Any) -> np.ndarray:
        """Exact containment of stored binary64 query points (no fuzzy boundary)."""
        query = _exact_points(_points(points))
        if self.factor is not None and self.center is not None:
            return np.array([v <= 1 for v in _exact_q(query, self.center, self.factor)], dtype=bool)
        if self.status == "DEGENERATE_POINT":
            p = _exact_points(self.unique_points)[0]
            return np.array([v == p for v in query], dtype=bool)
        if self.status == "DEGENERATE_SEGMENT":
            a, b = _exact_points(self.unique_points[[0, -1]])
            dx, dy = b[0]-a[0], b[1]-a[1]
            axis = 0 if dx else 1
            return np.array([(x-a[0])*dy == (y-a[1])*dx and
                0 <= ((x, y)[axis]-a[axis])/(b[axis]-a[axis]) <= 1 for x, y in query], dtype=bool)
        raise ValueError("no enclosing geometry available")

    def to_dict(self) -> dict:
        return {"contract": "ellipse-area-v1", "status": self.status, "converged": self.converged,
            "center": None if self.center is None else self.center.tolist(),
            "factor": None if self.factor is None else self.factor.tolist(),
            "area": self.area, "semiaxes": None if self.semiaxes is None else self.semiaxes.tolist(),
            "major_axis_length": self.major_axis_length, "iterations": self.iterations,
            "unique_points": self.unique_points.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "certificate": self.certificate}


def _failure(points, iterations, reason, weights=None) -> EllipseResult:
    if weights is not None and not np.isfinite(weights).all():
        weights = None
    return EllipseResult("NUMERICAL_FAILURE", None, None, None, None, None, iterations,
        points.copy(), None if weights is None else weights.copy(),
        {"proof_scope": PROOF_SCOPE, "reason": reason, "containment_exact": False,
         "area_approximation_certified": False, "exact_optimum": False})


def _degenerate(points, exact) -> EllipseResult | None:
    if len(points) == 1:
        return EllipseResult("DEGENERATE_POINT", points[0].copy(), None, 0.0,
            np.zeros(2), 0.0, 0, points.copy(), None,
            {"proof_scope": PROOF_SCOPE, "containment_exact": True, "affine_dimension": 0,
             "center_exact": [_rational(v) for v in exact[0]], "squared_length_exact": _rational(Fraction(0))})
    a, b = exact[0], exact[-1]
    dx, dy = b[0]-a[0], b[1]-a[1]
    if any((p[0]-a[0])*dy != (p[1]-a[1])*dx for p in exact):
        return None
    center_exact = [(a[i]+b[i])/2 for i in range(2)]
    center = np.array([float(v) for v in center_exact])
    squared_length = dx*dx+dy*dy
    try:
        length = math.hypot(float(dx), float(dy))
        if not math.isfinite(length):
            length = None
    except OverflowError:
        length = None
    axes = None if length is None or length/2 == 0 else np.array([length/2, 0.0])
    return EllipseResult("DEGENERATE_SEGMENT", center, None, 0.0,
        axes, length, 0, points.copy(), None,
        {"proof_scope": PROOF_SCOPE, "containment_exact": True, "affine_dimension": 1,
         "center_exact": [_rational(v) for v in center_exact], "squared_length_exact": _rational(squared_length),
         "center_display": "rounded binary64 midpoint; exact rational midpoint provided",
         "length_display": "rounded approximation" if length is not None else "outside binary64 range",
         "semiaxes_display": "rounded approximation" if axes is not None else "outside binary64 range; exact nonzero segment retained",
         "area_ratio": None})


def _certified_candidate(points, exact, weights, center, factor, iterations, tolerance,
                         stop_reason, solver_max_leverage, work) -> EllipseResult:
    if not np.isfinite(center).all() or not np.isfinite(factor).all():
        raise ArithmeticError("nonfinite candidate geometry")
    # Inflation is bounded. Every written-back float factor is rechecked exactly.
    inflation_passes = 0
    for attempt in range(4):
        q = _exact_q(exact, center, factor)
        maximum = max(q)
        if maximum <= 1:
            break
        if attempt == 3:
            raise ArithmeticError("containment inflation failed its exact recheck")
        divisor = _upper_sqrt(maximum)
        divisor = float(np.nextafter(divisor * (1 + 32*np.finfo(float).eps), math.inf))
        factor = factor/divisor
        inflation_passes += 1
    determinant = _f(factor[0, 0])*_f(factor[1, 1]) - _f(factor[0, 1])*_f(factor[1, 0])
    covariance_det = _covariance_det(exact, weights)
    if determinant == 0 or covariance_det <= 0:
        raise ArithmeticError("degenerate factor or dual covariance")
    ratio_squared = 1/(4*determinant**2*covariance_det)
    if ratio_squared < 1:
        raise ArithmeticError("inconsistent area lower bound and enclosing ellipse")
    accepted = ratio_squared <= (1+_f(tolerance))**2
    status = "APPROXIMATE_CERTIFIED" if accepted else (
        "MAX_ITERATIONS" if stop_reason in ("iteration_limit", "work_limit") else "NUMERICAL_FAILURE")
    log_area = math.log(math.pi) - math.log(abs(determinant.numerator)) + math.log(determinant.denominator)
    area = math.exp(log_area)
    singular = np.linalg.svd(factor, compute_uv=False)
    semiaxes = (1/singular)[::-1]
    major = 2*float(semiaxes[0])
    if area <= 0 or not math.isfinite(area) or not np.isfinite(semiaxes).all() or not math.isfinite(major):
        raise ArithmeticError("area or axis display outside binary64 range")
    certificate = {"proof_scope": PROOF_SCOPE, "containment_exact": True,
        "max_quadratic_exact": _rational(maximum), "det_factor_exact": _rational(determinant),
        "dual_covariance_det_exact": _rational(covariance_det),
        "area_ratio_squared": _rational(ratio_squared), "area_ratio_upper_bound": _upper_sqrt(ratio_squared),
        "requested_area_tolerance": tolerance, "area_approximation_certified": accepted,
        "exact_optimum": False, "center_coordinate_error_bound": None, "axis_error_bound": None,
        "stop_reason": "exact_area_certificate" if accepted else stop_reason,
        "solver_max_leverage": solver_max_leverage, "solver_point_update_work": work,
        "inflation_passes": inflation_passes,
        "weight_semantics": "stored nonnegative floats normalized exactly as rationals in the certificate",
        "area_display": "rounded pi/abs(det(B)); exact area-ratio certificate is independent of pi"}
    if status == "NUMERICAL_FAILURE":
        certificate["reason"] = "floating solver stop did not meet exact final area bound"
    return EllipseResult(status, center.copy(), factor.copy(), area, semiaxes, major,
        iterations, points.copy(), weights.copy(), certificate)


def minimum_enclosing_ellipse(points: Any, *, area_tolerance: float = 1e-3,
        max_iterations: int = 10_000, max_work: int = 10_000_000) -> EllipseResult:
    """Return an enclosing area approximation or an explicit non-success state.

    No point is dropped except exact duplicate coordinates. Convexity means that
    point containment also contains their convex hull. A tolerance bounds AREA,
    not centre/axis error and not physical accuracy.
    """
    p = _points(points)
    if isinstance(area_tolerance, (bool, np.bool_)) or not isinstance(area_tolerance, (int, float, np.floating)):
        raise EllipseInputError("area_tolerance must be a finite real number")
    try:
        tolerance = float(area_tolerance)
    except (OverflowError, ValueError) as exc:
        raise EllipseInputError("area_tolerance is outside binary64 range") from exc
    if not math.isfinite(tolerance) or not 1e-8 <= tolerance <= 0.25:
        raise EllipseInputError("area_tolerance must be in [1e-8,0.25]")
    if type(max_iterations) is not int or not 0 <= max_iterations <= MAX_ITERATIONS:
        raise EllipseInputError("max_iterations must be an integer in [0,100000]")
    if type(max_work) is not int or not 1 <= max_work <= MAX_WORK:
        raise EllipseInputError("max_work must be an integer in [1,100000000]")
    p = np.unique(p, axis=0)
    exact = _exact_points(p)
    degenerate = _degenerate(p, exact)
    if degenerate is not None:
        return degenerate
    iterations = 0
    weights = np.full(len(p), 1/len(p), dtype=float)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            origin = p[0].copy()
            difference = p-origin
            scale = float(np.max(np.abs(difference)))
            z = difference/scale
            z_mean = np.mean(z, axis=0)
            centered = z-z_mean
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            if singular[1] <= 8*np.finfo(float).eps*singular[0]:
                return _failure(p, 0, "exact rank two, but float64 whitening is ill-conditioned", weights)
            # Row-coordinate map: y=(z-z_mean) W. Whitening is numerical only;
            # final proof uses original world floats and includes its rounding.
            w = vt.T * (math.sqrt(len(p))/singular)
            y = centered @ w
            q = np.c_[y, np.ones(len(p))]
            effective_limit = min(max_iterations, max_work//len(p))
            while True:
                moment = (q.T*weights) @ q
                solved = np.linalg.solve(moment, q.T).T
                leverage = np.einsum("ij,ij->i", q, solved)
                j = int(np.argmax(leverage)); maximum = float(leverage[j])
                if not math.isfinite(maximum):
                    raise ArithmeticError("nonfinite leverage")
                if maximum <= 3+tolerance:  # reserve half the area gap for final rounding/inflation
                    stop_reason = "floating_optimality"
                    break
                if iterations >= effective_limit:
                    stop_reason = "work_limit" if effective_limit < max_iterations else "iteration_limit"
                    break
                step = (maximum-3)/(3*(maximum-1))
                if not 0 < step < 1:
                    raise ArithmeticError("invalid dual step")
                weights *= 1-step
                weights[j] += step
                weights /= math.fsum(float(v) for v in weights)
                iterations += 1
            mean = weights @ y
            centered_y = y-mean
            covariance = (centered_y.T*weights) @ centered_y
            chol = np.linalg.cholesky(covariance)
            b_work = np.linalg.solve(chol, np.eye(2))/math.sqrt(2)
            factor = b_work @ w.T / scale
            center = origin + scale*(z_mean + np.linalg.solve(w.T, mean))
            return _certified_candidate(p, exact, weights, center, factor, iterations, tolerance,
                stop_reason, maximum, iterations*len(p))
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
        return _failure(p, iterations, f"{type(exc).__name__}: {exc}", weights)
