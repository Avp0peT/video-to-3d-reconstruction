# Finite exact center API and CLI

`kozlov_lab.kozlov2025.exact_center.solve_exact_center(points, *, max_work=2000000, seconds=30)` takes the **complete** dictionary of 1–7 distinct labelled 2D points. Labels are Python integers; coordinates are integers or `fractions.Fraction`. Boolean and floating coordinates/labels are rejected. Coordinates use the existing 2048-bit input bound. No points are sampled or dropped.

The search order is point/collinear segment, triangular hull, a three-contact support, rational five-contact supports, and algebraic four-contact supports. Every constructor gets the remaining work and time, and its reported work is deducted. Final independent verification also consumes that same budget. Time checks are cooperative rather than a hard process timeout. Missing certificates and exhausted limits remain `PARTIAL_UNKNOWN`; they do not prove that an enclosing ellipse does not exist and do not certify arbitrary input counts or bit sizes.

```python
from fractions import Fraction
from kozlov_lab.kozlov2025.exact_center import (
    solve_exact_center, verify_exact_center_result,
    encode_exact, decode_exact,
)

points = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (2, Fraction(3, 2))}
result = solve_exact_center(points)
assert result['status'] == 'EXACT_OPTIMUM_COMPLETE_INPUT'
assert verify_exact_center_result(points, result)
```

Results have `schema: exact-center-v1`, `declared_labels`, `source_coordinate_sha256`, `profile`, `number_context`, `center`, `major_axis_length`, and the full underlying `certificate`. `attempts` and `work_used` describe bounded construction. Rational profiles use `number_context: {domain: rational}` and Fraction values. Four-contact results use `number_context: {domain: real-algebraic, algebraic_root: ...}`; each coordinate is the serialized exact numerator/denominator polynomial expression at that root. The true cubic example above remains algebraic, without converting its center to a decimal.

For positive definite ellipses, `major_axis_length` is the exact formula `sqrt(2*(trace+sqrt(discriminant)))`, with exact trace and discriminant. It is a symbolic length, not a floating measurement. The complete input source binding and all containment/moment identities remain in the certificate.

Point and collinear inputs return `EXACT_DEGENERATE_COMPLETE_INPUT`, with `profile: point` or `segment`. Their center is the point itself or the midpoint of the extreme enclosing segment; length is `sqrt(length_squared)`. This is a canonical minimum enclosing point/segment result. It does not assert existence or a unique center of a positive definite minimum-area ellipse for degenerate data.

`verify_exact_center_result(points, result, *, max_work=..., seconds=...)` returns a boolean and invokes only certificate verification, never candidate constructors. `assess_exact_center_result` distinguishes `VERIFIED`, `INVALID`, and `UNKNOWN`, and reports `work_used`; use it when another bounded algorithm must charge verification. A boolean false alone cannot distinguish invalid evidence from exhausted verification budget.

JSON transport must use `encode_exact` and `decode_exact`. Fractions, tuples, and integer-keyed maps have explicit `$exact` tags (`rational`, `tuple`, `integer-map`). This preserves point identities and the existing certificate sequence contracts. Do not call ordinary `json.dumps` on an untagged certificate: Python silently converts integer object keys to strings. Decoding rejects duplicate integer-map identities, floating values, nonreduced rationals and malformed tags. Trees and normalized rational values are bounded; the decoder additionally limits rational decimal strings to 4300 characters.

The standalone `scripts/solve_exact_center.py` accepts this strict input:

```json
{
  "schema": "exact-center-input-v1",
  "points": [
    {"label": 0, "xy": [0, 0]},
    {"label": 1, "xy": [1, 0]},
    {"label": 2, "xy": [0, 1]},
    {"label": 3, "xy": [2, {"numerator": "3", "denominator": "2"}]}
  ]
}
```

With a configured Python interpreter and `src` available, run:

```text
python -B -X utf8 scripts/solve_exact_center.py --input points.json --output new-result.json
python -B -X utf8 scripts/solve_exact_center.py --input points.json --verify new-result.json --output new-check.json
```

The script adds the repository `src` path itself and uses only the standard library. Output is `exact-center-cli-result-v1`, wrapping the tagged API result, the input byte hash, resolved input path, and CLI arguments. Construction verifies a tagged JSON roundtrip before publishing an exact success. Verification mode independently checks a prior result against the full input and its byte hash. Both modes require a new output path and refuse overwrite. Inputs are limited to 4 MiB and reject duplicate JSON keys, duplicate labels, noncanonical or nonreduced rationals, booleans, floats, and NaN/Infinity.

Exit codes are 0 for exact/verified results, 1 for bounded unknown or failed verification, and 2 for malformed input/evidence or filesystem errors. No native process, shell, model, GPU, network, public-init change, research-page import, or production reconstruction is required.
