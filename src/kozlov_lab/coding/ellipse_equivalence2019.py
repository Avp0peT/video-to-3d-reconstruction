"""S-2019-CODES pp58-59: bounded search of complete indexed ellipse codes.

The paper defines the bijection predicate; exhaustive search and rejection
certificates are project implementations. No affine-anchor pruning is used.
"""
from fractions import Fraction as F
from itertools import permutations
import math

from ..registration.fixed_orientation import _Budget, _Exhausted, _label, _fraction
from .ellipse_code2019 import (
    _input, _key, _remaining, _tree, build_ellipse_code2019,
    assess_ellipse_code2019,
)
from .environment_code import _same

SCHEMA = 'ellipse-equivalence-2019-v1'
SOURCE = 'S-2019-CODES:58-59'
CODEC_SCHEMA = 'ellipse-equivalence-2019-json-v1'
_CERT_FIELDS = {'schema', 'source', 'source_code', 'target_code', 'verdict', 'mapping',
                'ratio_ledger', 'rejection_chunks', 'affine_witness', 'theorem_domain'}
_CORE_FIELDS = _CERT_FIELDS - {'ratio_ledger', 'rejection_chunks'}


def _charge(result, budget, expected):
    budget.spend(result['work_used'])
    if result['status'] == 'UNKNOWN':
        raise _Exhausted('nested ellipse code budget exhausted')
    if result['status'] != expected:
        raise ValueError('invalid complete ellipse code: ' + result['status'])
    return result


def _mapped_key(key, mapping):
    return _key(tuple(sorted(mapping[int(i)] for i in key.split(','))))


def _comparison_rows(left, right, mapping, budget, stop_at_mismatch=False):
    rows = []
    for key, value in left['ellipse_ratios'].items():
        budget.spend(8)
        numerator, denominator = key.split('/')
        mapped = _mapped_key(numerator, mapping) + '/' + _mapped_key(denominator, mapping)
        target_value = right['ellipse_ratios'][mapped]
        row = (key, mapped, value, target_value)
        if stop_at_mismatch and value != target_value:
            return row
        rows.append(row)
    return None if stop_at_mismatch else tuple(rows)


def _affine_witness(a, b, mapping, budget):
    """Optional geometric witness for this exact mapping; no theorem is used."""
    i, j, k = sorted(a)[:3]
    p, q, r = a[i], a[j], a[k]
    u, v, w = b[mapping[i]], b[mapping[j]], b[mapping[k]]
    dx, dy, ex, ey = q[0]-p[0], q[1]-p[1], r[0]-p[0], r[1]-p[1]
    determinant = dx*ey-ex*dy
    matrix = tuple(((v[t]-u[t])*ey-(w[t]-u[t])*dy,
                    (w[t]-u[t])*dx-(v[t]-u[t])*ex) for t in (0, 1))
    matrix = tuple(tuple(x/determinant for x in row) for row in matrix)
    offset = tuple(u[t]-sum(matrix[t][s]*p[s] for s in (0, 1)) for t in (0, 1))
    for label, point in a.items():
        budget.spend(20)
        if tuple(sum(matrix[t][s]*point[s] for s in (0, 1))+offset[t] for t in (0, 1)) != b[mapping[label]]:
            return None
    return dict(matrix=matrix, offset=offset)


def _verify_geometry(a, b, mapping, witness, budget):
    if witness is None:
        return
    if type(witness) is not dict or set(witness) != {'matrix', 'offset'}:
        raise ValueError('affine witness fields')
    matrix, offset = witness['matrix'], witness['offset']
    if (type(matrix) is not tuple or len(matrix) != 2
            or any(type(row) is not tuple or len(row) != 2 for row in matrix)
            or type(offset) is not tuple or len(offset) != 2):
        raise ValueError('affine witness shape')
    if any(type(v) not in (int, F) for row in matrix for v in row) or any(type(v) not in (int, F) for v in offset):
        raise ValueError('exact affine scalars required')
    if matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0] == 0:
        raise ValueError('affine witness must be invertible')
    for label, point in a.items():
        budget.spend(20)
        # Independent homogeneous point substitution, without the constructor.
        x, y = point
        if (matrix[0][0]*x+matrix[0][1]*y+offset[0],
                matrix[1][0]*x+matrix[1][1]*y+offset[1]) != b[mapping[label]]:
            raise ValueError('affine witness fails complete point binding')


def _verify(a, b, certificate, budget):
    _tree(certificate, budget)
    fields = {'schema', 'source', 'source_code', 'target_code', 'verdict', 'mapping',
              'ratio_ledger', 'rejection_chunks', 'affine_witness', 'theorem_domain'}
    if type(certificate) is not dict or set(certificate) != fields:
        raise ValueError('equivalence certificate fields')
    c = certificate
    if c['schema'] != SCHEMA or c['source'] != SOURCE:
        raise ValueError('equivalence source/schema')
    for points, code in ((a, c['source_code']), (b, c['target_code'])):
        _charge(assess_ellipse_code2019(points, code, **_remaining(budget)), budget, 'VERIFIED')
    left, right = c['source_code'], c['target_code']
    domain = left['area_code']['plane_domain'] and right['area_code']['plane_domain']
    if type(c['theorem_domain']) is not bool or c['theorem_domain'] is not domain:
        raise ValueError('paper plane-domain binding')
    if c['verdict'] == 'ELLIPSE_EQUIVALENT':
        if type(c['mapping']) is not tuple:
            raise ValueError('mapping tuple required')
        for pair in c['mapping']:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('mapping pair shape')
            _label(pair[0]); _label(pair[1])
        mapping = dict(c['mapping'])
        if (len(mapping) != len(c['mapping']) or set(mapping) != set(a)
                or len(a) != len(b) or set(mapping.values()) != set(b)
                or not _same(c['mapping'], tuple(sorted(mapping.items())))):
            raise ValueError('complete unique source-to-target bijection required')
        if c['rejection_chunks'] != ():
            raise ValueError('positive certificate has no exhaustion claim')
        ledger = c['ratio_ledger']
        if type(ledger) is not tuple or len(ledger) != len(left['ellipse_ratios']):
            raise ValueError('complete indexed ratio ledger required')
        # Independently enumerate the source indices, rather than invoking the
        # search/comparison builder. Both codes already have complete verification.
        for row, (key, value) in zip(ledger, left['ellipse_ratios'].items()):
            budget.spend(12)
            numerator, denominator = key.split('/')
            def mapped(triple):
                return ','.join(str(v) for v in sorted(mapping[int(x)] for x in triple.split(',')))
            target_key = mapped(numerator) + '/' + mapped(denominator)
            expected = (key, target_key, value, right['ellipse_ratios'][target_key])
            if not _same(row, expected) or expected[2] != expected[3]:
                raise ValueError('indexed ellipse equality mismatch')
        _verify_geometry(a, b, mapping, c['affine_witness'], budget)
    elif c['verdict'] == 'NOT_ELLIPSE_EQUIVALENT':
        if c['mapping'] != () or c['ratio_ledger'] != () or c['affine_witness'] is not None:
            raise ValueError('negative certificate cannot assert a mapping')
        chunks = c['rejection_chunks']
        if type(chunks) is not tuple or any(type(chunk) is not tuple or not 1 <= len(chunk) <= 256 for chunk in chunks):
            raise ValueError('rejection chunk format')
        rejected = tuple(row for chunk in chunks for row in chunk)
        count = math.factorial(len(a)) if len(a) == len(b) else 0
        if len(rejected) != count:
            raise ValueError('negative requires complete bijection coverage')
        if len(a) == len(b):
            labels = tuple(sorted(a))
            for row, image in zip(rejected, permutations(sorted(b))):
                budget.spend(12)
                if type(row) is not tuple or len(row) != 2 or not _same(row[0], image):
                    raise ValueError('rejection ordering/coverage mismatch')
                proof = row[1]
                if type(proof) is not tuple or len(proof) != 4:
                    raise ValueError('rejection ratio witness shape')
                source_key, target_key, source_value, target_value = proof
                if type(source_key) is not str or source_key not in left['ellipse_ratios']:
                    raise ValueError('unknown rejection source index')
                mapping = dict(zip(labels, image))
                numerator, denominator = source_key.split('/')
                expected_target = _mapped_key(numerator, mapping) + '/' + _mapped_key(denominator, mapping)
                expected = (source_key, expected_target, left['ellipse_ratios'][source_key], right['ellipse_ratios'][expected_target])
                if not _same(proof, expected) or source_value == target_value:
                    raise ValueError('rejection must prove a genuine exact ratio inequality')
    else:
        raise ValueError('invalid equivalence verdict')
    return c


def _summary(c):
    equivalent = c['verdict'] == 'ELLIPSE_EQUIVALENT'
    return dict(code_equivalent=equivalent, theorem_domain=c['theorem_domain'],
                theorem_conclusion=('AFFINE_EQUIVALENT' if equivalent else 'NOT_AFFINE_EQUIVALENT')
                    if c['theorem_domain'] else 'NOT_APPLICABLE',
                affine_mapping_status='WITNESS_VERIFIED' if c['affine_witness'] is not None else 'NOT_ESTABLISHED')


def search_ellipse_equivalence2019(source, target, *, max_work=2000000, seconds=30):
    """Decide the paper's code predicate on complete strict 3..8-point inputs.

    Unequal cardinalities give a direct negative. Same-sized negatives require
    all n! mappings and a separate inequality witness for each. Cancellation
    returns UNKNOWN without a partial certificate or a negative conclusion.
    """
    budget = None
    try:
        budget = _Budget(max_work, seconds); budget.spend()
        a, b = _input(source), _input(target)
        left = _charge(build_ellipse_code2019(a, **_remaining(budget)), budget, 'EXACT_ELLIPSE_CODE')['certificate']
        right = _charge(build_ellipse_code2019(b, **_remaining(budget)), budget, 'EXACT_ELLIPSE_CODE')['certificate']
        c = dict(schema=SCHEMA, source=SOURCE, source_code=left, target_code=right,
                 verdict='NOT_ELLIPSE_EQUIVALENT', mapping=(), ratio_ledger=(),
                 rejection_chunks=(), affine_witness=None,
                 theorem_domain=left['area_code']['plane_domain'] and right['area_code']['plane_domain'])
        rejected = []
        if len(a) == len(b):
            labels = tuple(sorted(a))
            for image in permutations(sorted(b)):
                budget.spend()
                mapping = dict(zip(labels, image))
                mismatch = _comparison_rows(left, right, mapping, budget, stop_at_mismatch=True)
                if mismatch is None:
                    c.update(verdict='ELLIPSE_EQUIVALENT', mapping=tuple(sorted(mapping.items())),
                             ratio_ledger=_comparison_rows(left, right, mapping, budget),
                             affine_witness=_affine_witness(a, b, mapping, budget))
                    break
                rejected.append((image, mismatch))
            else:
                c['rejection_chunks'] = tuple(tuple(rejected[i:i+256]) for i in range(0, len(rejected), 256))
        _verify(a, b, c, budget)
        return dict(status=c['verdict'], certificate=c, work_used=budget.used, **_summary(c))
    except _Exhausted as exc:
        return dict(status='UNKNOWN', reason=str(exc), certificate=None, work_used=budget.used,
                    code_equivalent=None, theorem_conclusion='NOT_DECIDED', affine_mapping_status='NOT_ESTABLISHED')
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return dict(status='UNSUPPORTED', reason=str(exc), certificate=None,
                    work_used=budget.used if budget else 0, code_equivalent=None,
                    theorem_conclusion='NOT_DECIDED', affine_mapping_status='NOT_ESTABLISHED')


def assess_ellipse_equivalence2019(source, target, certificate, *, max_work=2000000, seconds=30):
    """Independent certificate readback; never invokes the search constructor."""
    budget = None
    try:
        budget = _Budget(max_work, seconds); budget.spend()
        c = _verify(_input(source), _input(target), certificate, budget)
        return dict(status='VERIFIED', verdict=c['verdict'], work_used=budget.used, **_summary(c))
    except _Exhausted as exc:
        return dict(status='UNKNOWN', reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return dict(status='INVALID', reason=str(exc), work_used=budget.used if budget else 0)


def _codec_structure(c, budget):
    """Bounds and exact field types only; not a mathematical certificate check."""
    if type(c) is not dict or set(c) != _CERT_FIELDS or c['schema'] != SCHEMA or c['source'] != SOURCE:
        raise ValueError('codec certificate fields/source/schema')
    if c['verdict'] not in ('ELLIPSE_EQUIVALENT', 'NOT_ELLIPSE_EQUIVALENT'):
        raise ValueError('codec verdict')
    if type(c['theorem_domain']) is not bool:
        raise ValueError('codec theorem_domain must be bool')
    allowed_bool_paths = {('theorem_domain',)}
    ellipse_fields = {'schema', 'source', 'domain_policy', 'area_code', 'ellipses',
                      'ellipse_ratios', 'geometric_meaning'}
    area_fields = {'schema', 'source', 'environment_points', 'labels', 'double_areas',
                   'zero_denominators', 'ratios', 'code_defined', 'domain', 'plane_domain'}
    for name in ('source_code', 'target_code'):
        code = c[name]
        if type(code) is not dict or set(code) != ellipse_fields:
            raise ValueError('codec complete ellipse fields')
        area = code['area_code']
        if type(area) is not dict or set(area) != area_fields:
            raise ValueError('codec complete area fields')
        for key in ('code_defined', 'plane_domain'):
            if type(area[key]) is not bool:
                raise ValueError('codec area bool fields')
            allowed_bool_paths.add((name, 'area_code', key))
        for owner, key, bound in ((code, 'ellipses', 56), (code, 'ellipse_ratios', 3080),
                                  (area, 'double_areas', 56), (area, 'ratios', 3080)):
            if type(owner[key]) is not dict or len(owner[key]) > bound:
                raise ValueError('codec complete code container bound')
    def booleans(value, path=()):
        budget.spend()
        if len(path) > 24:
            raise ValueError('codec native tree depth bound')
        if type(value) is bool:
            if path not in allowed_bool_paths:
                raise ValueError('codec boolean in numeric/structural position')
        elif type(value) is dict:
            if len(value) > 4096:
                raise ValueError('codec native mapping bound')
            for key, child in value.items():
                if type(key) is not str or len(key) > 256:
                    raise ValueError('codec string certificate keys required')
                booleans(child, path + (key,))
        elif type(value) in (tuple, list):
            if len(value) > 4096:
                raise ValueError('codec native sequence bound')
            for index, child in enumerate(value):
                booleans(child, path + (index,))
        elif type(value) in (int, F):
            _fraction(value, 16384)
        elif type(value) is str:
            if len(value) > 256:
                raise ValueError('codec native string bound')
        elif value is not None:
            raise ValueError('codec exact native types required')
    booleans(c)
    mapping = c['mapping']
    if type(mapping) is not tuple or len(mapping) > 8:
        raise ValueError('codec mapping bound')
    for pair in mapping:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError('codec mapping pair')
        _label(pair[0]); _label(pair[1])
    def ratio(row):
        if (type(row) is not tuple or len(row) != 4
                or any(type(v) is not str for v in row[:2])
                or any(type(v) not in (int, F) for v in row[2:])):
            raise ValueError('codec exact ratio row')
    ledger = c['ratio_ledger']
    if type(ledger) is not tuple or len(ledger) > 3080:
        raise ValueError('codec ratio count bound')
    for row in ledger:
        ratio(row)
    chunks = c['rejection_chunks']
    if type(chunks) is not tuple or len(chunks) > 158:
        raise ValueError('codec rejection chunk bound')
    total = 0
    for chunk in chunks:
        if type(chunk) is not tuple or not 1 <= len(chunk) <= 256:
            raise ValueError('codec rejection chunk size')
        total += len(chunk)
        for row in chunk:
            if type(row) is not tuple or len(row) != 2 or type(row[0]) is not tuple or not 3 <= len(row[0]) <= 8:
                raise ValueError('codec rejection row')
            for label in row[0]:
                _label(label)
            ratio(row[1])
    if total > 40320:
        raise ValueError('codec rejection count bound')


def _json_block(value, budget):
    """Preflight one JSON block before the unchanged 20k-node exact decoder."""
    remaining = [200000]
    def visit(item, depth=0):
        budget.spend(); remaining[0] -= 1
        if remaining[0] < 0 or depth > 32:
            raise ValueError('codec raw JSON block bound')
        if type(item) is dict:
            if len(item) > 4096 or any(type(k) is not str or len(k) > 256 for k in item):
                raise ValueError('codec JSON object bound')
            for child in item.values():
                visit(child, depth+1)
        elif type(item) is list:
            if len(item) > 4096:
                raise ValueError('codec JSON array bound')
            for child in item:
                visit(child, depth+1)
        elif type(item) is str:
            if len(item) > 4300:
                raise ValueError('codec JSON string bound')
        elif type(item) is int:
            if abs(item).bit_length() > 16384:
                raise ValueError('codec JSON integer bound')
        elif type(item) is not bool and item is not None:
            raise ValueError('codec JSON forbids floats/non-JSON objects')
    visit(value)


def encode_ellipse_equivalence2019(certificate, *, max_work=2000000, seconds=30):
    """Return bounded JSON envelope; raises on malformed data or budget expiry.

    Serialization never establishes mathematical validity. Each core field and
    256-row chunk uses the existing exact codec independently.
    """
    from ..kozlov2025.exact_center import encode_exact, decode_exact
    budget = _Budget(max_work, seconds)
    _codec_structure(certificate, budget)
    def block(value):
        encoded = encode_exact(value)
        _json_block(encoded, budget)
        # Keep the existing per-block 20,000 decoded-node limit on encoding too.
        decode_exact(encoded)
        budget.spend(0)
        return encoded
    envelope = dict(schema=CODEC_SCHEMA,
        core={key: block(certificate[key]) for key in sorted(_CORE_FIELDS)},
        ratio_chunks=[block(certificate['ratio_ledger'][i:i+256])
                      for i in range(0, len(certificate['ratio_ledger']), 256)],
        rejection_chunks=[block(chunk) for chunk in certificate['rejection_chunks']])
    return envelope


def decode_ellipse_equivalence2019(envelope, *, max_work=2000000, seconds=30):
    """Parse exact bounded structure only; call assess separately for validity.

    JSON duplicate-object-key rejection remains the caller's parsing policy.
    This API takes an already parsed JSON object, never raw JSON text.
    """
    from ..kozlov2025.exact_center import decode_exact
    budget = _Budget(max_work, seconds); budget.spend()
    if (type(envelope) is not dict or set(envelope) != {'schema', 'core', 'ratio_chunks', 'rejection_chunks'}
            or envelope['schema'] != CODEC_SCHEMA):
        raise ValueError('codec envelope fields/schema')
    core = envelope['core']
    if type(core) is not dict or set(core) != _CORE_FIELDS:
        raise ValueError('codec complete core fields')
    for key, bound in (('ratio_chunks', 13), ('rejection_chunks', 158)):
        if type(envelope[key]) is not list or len(envelope[key]) > bound:
            raise ValueError('codec envelope chunk count bound')
    def block(value):
        _json_block(value, budget)
        decoded = decode_exact(value)
        budget.spend(0)
        return decoded
    c = {key: block(core[key]) for key in sorted(_CORE_FIELDS)}
    ratios, rejections = [], []
    for name, destination in (('ratio_chunks', ratios), ('rejection_chunks', rejections)):
        for encoded in envelope[name]:
            chunk = block(encoded)
            if type(chunk) is not tuple or not 1 <= len(chunk) <= 256:
                raise ValueError('codec decoded chunk size')
            destination.append(chunk)
    c['ratio_ledger'] = tuple(row for chunk in ratios for row in chunk)
    c['rejection_chunks'] = tuple(rejections)
    _codec_structure(c, budget)
    return c
