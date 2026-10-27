"""Bounded exact sufficient certificate for complete 4..7 centre alignment."""
from fractions import Fraction as F
from itertools import combinations, permutations
from math import factorial
from .affine_subimage import _rows, _sub, _det, _solve, _linear
from .zero_error_fits import _budget
from .independent_cover import BudgetExceeded

SCHEMA = 'circuit-center-alignment-v1'
SUCCESS = 'GLOBAL_OPTIMAL_STABLE_SUFFICIENT'


def _inputs(target, source):
    x, y = _rows(target, 7), _rows(source, 7)
    if not 4 <= len(x) == len(y) <= 7:
        raise ValueError('equal complete cardinalities 4..7 required')
    for rows in (x, y):
        if not any(_det(_sub(b[1], a[1]), _sub(c[1], a[1]))
                   for a, b, c in combinations(rows, 3)):
            raise ValueError('both affine ranks must be two')
    return x, y


def _relation(y, circuit):
    points = [y[i][1] for i in circuit]
    c = []
    for i in range(4):
        a, b, d = [q for j, q in enumerate(points) if j != i]
        c.append(F((-1)**i)*_det(_sub(b, a), _sub(d, a)))
    if not all(c):
        raise ValueError('circuit must have every triple noncollinear')
    return tuple(q/c[0] for q in c)


def _vector(x, p, circuit, c):
    s = sum(abs(q) for q in c)
    return tuple(-sum(c[j]*x[p[i]][1][k] for j, i in enumerate(circuit))/s
                 for k in range(2))


def _energy(v):
    return sum(q*q for q in v)


def _radius(x, epsilon):
    a, b, c = next(t for t in combinations(x, 3)
                   if _det(_sub(t[1][1], t[0][1]), _sub(t[2][1], t[0][1])))
    u, v = _sub(b[1], a[1]), _sub(c[1], a[1])
    d = _det(u, v)
    inverse = ((v[1]/d, -v[0]/d), (-u[1]/d, u[0]/d))
    k = max(sum(abs(inverse[i][j]) for i in range(2)) for j in range(2))
    return epsilon/(4*max(F(1), k))


def _epsilon(ledger, e):
    return min([F(1, 4)] + [(z['lower_bound_squared']-e)/
               (8*(z['lower_bound_squared']+e)) for z in ledger
               if z['lower_bound_squared'] > e])


def _tree(value, budget, depth=0):
    budget.spend()
    if depth > 32:
        raise ValueError('certificate nesting bound')
    if type(value) in (int, F):
        q = F(value)
        if max(q.numerator.bit_length(), q.denominator.bit_length()) > 16384:
            raise BudgetExceeded('stored rational bit bound')
    elif type(value) is tuple:
        if len(value) > 5040:
            raise ValueError('certificate tuple bound')
        for q in value:
            _tree(q, budget, depth+1)
    elif type(value) is dict:
        if len(value) > 32 or any(type(k) is not str for k in value):
            raise ValueError('certificate dictionary bound')
        for q in value.values():
            _tree(q, budget, depth+1)
    elif type(value) is str:
        if len(value) > 200:
            raise ValueError('certificate string bound')
    else:
        raise ValueError('strict exact types required; bool rejected')


def _fractions(v, size):
    return type(v) is tuple and len(v) == size and all(type(q) is F for q in v)


def _verify(x, y, c, budget):
    # No producer, candidate construction, or affine solver is invoked here.
    _tree(c, budget)
    fields={'schema','target','source','ledger','chosen','matrix','translation','alpha',
            'error_squared','epsilon','epsilon1','epsilon_point_displacement','classification'}
    if type(c) is not dict or set(c)!=fields:
        raise ValueError('strict complete alignment certificate fields required')
    if c['schema'] != SCHEMA or c['target'] != x or c['source'] != y:
        raise ValueError('complete input binding mismatch')
    for rows in (c['target'], c['source']):
        if type(rows) is not tuple or any(type(r) is not tuple or len(r) != 2 or type(r[0]) is not int or not _fractions(r[1], 2) for r in rows):
            raise ValueError('typed canonical input binding required')
    ledger = c['ledger']; n = len(y)
    if type(ledger) is not tuple or len(ledger) != factorial(n):
        raise ValueError('complete permutation ledger required')
    for p, z in zip(permutations(range(n)), ledger):
        budget.spend(40)
        if type(z) is not dict or set(z)!={'permutation','circuit','vector','lower_bound_squared'}:
            raise ValueError('strict circuit lower-bound row fields required')
        if z['permutation'] != p or any(type(i) is not int for i in z['permutation']):
            raise ValueError('canonical unique permutation ledger required')
        ids = z['circuit']
        if (type(ids) is not tuple or len(ids) != 4 or
            any(type(i) is not int or not 0 <= i < n for i in ids) or
            tuple(sorted(set(ids))) != ids):
            raise ValueError('canonical circuit indices required')
        relation = _relation(y, ids)
        v = z['vector']
        if not _fractions(v, 2) or v != _vector(x, p, ids, relation):
            raise ValueError('circuit lower bound vector mismatch')
        if type(z['lower_bound_squared']) is not F or z['lower_bound_squared'] != _energy(v):
            raise ValueError('circuit lower bound energy mismatch')
    e = min(z['lower_bound_squared'] for z in ledger)
    chosen = c['chosen']
    if type(chosen) is not int or not 0 <= chosen < len(ledger):
        raise ValueError('chosen index invalid')
    z = ledger[chosen]; v = z['vector']
    if type(c['error_squared']) is not F or c['error_squared'] != e or z['lower_bound_squared'] != e:
        raise ValueError('global lower bound mismatch')
    if any(q['vector'] not in (v, tuple(-a for a in v)) for q in ledger if q['lower_bound_squared'] == e):
        raise ValueError('minimum directions not stable')
    m, t, alpha = c['matrix'], c['translation'], c['alpha']
    if type(m) is not tuple or len(m) != 2 or not all(_fractions(r, 2) for r in m) or not _fractions(t, 2) or not _fractions(alpha, n):
        raise ValueError('typed complete affine witness required')
    if not m[0][0]*m[1][1]-m[0][1]*m[1][0]:
        raise ValueError('invertible witness required')
    for i, (_, point) in enumerate(y):
        budget.spend(16)
        r = tuple(m[k][0]*point[0]+m[k][1]*point[1]+t[k]-x[z['permutation'][i]][1][k] for k in range(2))
        if abs(alpha[i]) > 1 or r != tuple(alpha[i]*q for q in v) or (not any(v) and alpha[i] != 0):
            raise ValueError('complete residual/alpha substitution failed')
    eps = _epsilon(ledger, e)
    if (type(c['epsilon']) is not F or c['epsilon'] != eps or
        type(c['epsilon1']) is not F or c['epsilon1'] != 1 or
        type(c['epsilon_point_displacement']) is not F or
        c['epsilon_point_displacement'] != _radius(x, eps) or c['classification'] != SUCCESS):
        raise ValueError('stability fields mismatch')
    budget.spend(0)
    return SUCCESS


def align_circuit_centers(target, source, *, max_work=2000000, seconds=30):
    budget = None; counts = dict(permutations_completed=0, circuits_evaluated=0, candidates_checked=0)
    try:
        budget = _budget(max_work, seconds); budget.spend()
        x, y = _inputs(target, source)
        circuits = []
        for ids in combinations(range(len(y)), 4):
            budget.spend(8)
            try:
                circuits.append((ids, _relation(y, ids)))
            except ValueError:
                pass
        if not circuits:
            return dict(status='UNSUPPORTED', reason='no admissible four-point source circuit', work_used=budget.used, **counts)
        ledger = []
        for p in permutations(range(len(y))):
            best = None
            for ids, relation in circuits:
                budget.spend(20); counts['circuits_evaluated'] += 1
                v = _vector(x, p, ids, relation); e = _energy(v)
                if best is None or e > best['lower_bound_squared']:
                    best = dict(permutation=p, circuit=ids, vector=v, lower_bound_squared=e)
            ledger.append(best); counts['permutations_completed'] += 1
        ledger = tuple(ledger); e = min(z['lower_bound_squared'] for z in ledger)
        for chosen, z in enumerate(ledger):
            if z['lower_bound_squared'] != e:
                continue
            budget.spend(80); counts['candidates_checked'] += 1
            ids = z['circuit']; relation = _relation(y, ids); v = z['vector']; p = z['permutation']
            if any(q['vector'] not in (v, tuple(-a for a in v)) for q in ledger if q['lower_bound_squared'] == e):
                continue
            images = [tuple(x[p[i]][1][k]+(1 if relation[j] > 0 else -1)*v[k] for k in range(2)) for j, i in enumerate(ids)]
            a, b, d = [y[i][1] for i in ids[:3]]
            m = _solve(_sub(b, a), _sub(d, a), _sub(images[1], images[0]), _sub(images[2], images[0]))
            if not m[0][0]*m[1][1]-m[0][1]*m[1][0]:
                continue
            t = _sub(images[0], _linear(m, a)); alpha = []
            for i, (_, point) in enumerate(y):
                budget.spend(16)
                r = _sub(tuple(q+t[k] for k, q in enumerate(_linear(m, point))), x[p[i]][1])
                k = 0 if v[0] else 1
                q = r[k]/v[k] if any(v) else F(0)
                if abs(q) > 1 or r != tuple(q*h for h in v):
                    break
                alpha.append(q)
            if len(alpha) != len(y):
                continue
            eps = _epsilon(ledger, e)
            c = dict(schema=SCHEMA, target=x, source=y, ledger=ledger, chosen=chosen, matrix=m, translation=t, alpha=tuple(alpha), error_squared=e, epsilon=eps, epsilon1=F(1), epsilon_point_displacement=_radius(x, eps), classification=SUCCESS)
            _verify(x, y, c, budget)
            return dict(status=SUCCESS, certificate=c, work_used=budget.used, **counts)
        return dict(status='UNKNOWN', reason='sufficient attainment/stability conditions not established', work_used=budget.used, **counts)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN', reason=str(exc), work_used=budget.used if budget else 0, **counts)
    except (ValueError, TypeError, KeyError, ArithmeticError, IndexError) as exc:
        return dict(status='UNSUPPORTED', reason=str(exc), work_used=budget.used if budget else 0, **counts)


def assess_circuit_alignment_certificate(target, source, certificate, *, max_work=2000000, seconds=30):
    """Independent full permutation lower bound and affine substitution check."""
    budget = None
    try:
        budget = _budget(max_work, seconds); budget.spend()
        x, y = _inputs(target, source)
        classification = _verify(x, y, certificate, budget)
        return dict(status='VERIFIED', classification=classification, work_used=budget.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN', reason=str(exc), work_used=budget.used if budget else 0)
    except (ValueError, TypeError, KeyError, ArithmeticError, IndexError, AttributeError) as exc:
        return dict(status='INVALID', reason=str(exc), work_used=budget.used if budget else 0)
