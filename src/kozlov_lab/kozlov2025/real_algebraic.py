"""Bounded exact real algebraic arithmetic, using only rational Sturm methods.

Polynomials are low-to-high Fraction tuples. Bounds are machine refusal gates,
not claims that every bounded input can be decided within the shared budget.
"""
from fractions import Fraction as F
from .independent_cover import Budget, BudgetExceeded

MAX_BITS = 16384
MAX_LENGTH = 13
MAX_STEPS = 4096


def _charge(budget, n=1):
    if budget is not None:
        budget.spend(n)


def _q(x):
    if type(x) not in (int, F):
        raise ValueError('integer or Fraction required')
    x = F(x)
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > MAX_BITS:
        raise BudgetExceeded('rational bit bound exceeded')
    return x


def trim(p, budget=None):
    _charge(budget)
    if type(p) not in (tuple, list) or not 1 <= len(p) <= MAX_LENGTH:
        raise ValueError('bounded nonempty polynomial required')
    a = [_q(x) for x in p]
    while len(a) > 1 and not a[-1]:
        a.pop()
    return tuple(a)


def add(a, b, budget=None):
    a, b = trim(a, budget), trim(b, budget)
    return trim(tuple((a[i] if i < len(a) else 0)+(b[i] if i < len(b) else 0)
                      for i in range(max(len(a), len(b)))), budget)


def sub(a, b, budget=None):
    return add(a, tuple(-x for x in trim(b, budget)), budget)


def mul(a, b, budget=None):
    a, b = trim(a, budget), trim(b, budget)
    _charge(budget, len(a)*len(b))
    if len(a)+len(b)-1 > MAX_LENGTH:
        raise BudgetExceeded('polynomial degree bound exceeded')
    c = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] = _q(c[i+j]+x*y)
    return trim(c, budget)


def divrem(a, b, budget=None):
    a, b = trim(a, budget), trim(b, budget)
    if b == (0,):
        raise ZeroDivisionError('zero polynomial')
    q = [F(0)]*max(1, len(a)-len(b)+1)
    r = list(a)
    while tuple(r) != (0,) and len(r) >= len(b):
        _charge(budget, len(b))
        k, c = len(r)-len(b), _q(r[-1]/b[-1])
        q[k] = c
        for j, x in enumerate(b):
            r[k+j] = _q(r[k+j]-c*x)
        r = list(trim(r, budget))
    return trim(q, budget), tuple(r)


def gcd(a, b, budget=None):
    a, b = trim(a, budget), trim(b, budget)
    while b != (0,):
        a, b = b, divrem(a, b, budget)[1]
    return tuple(_q(x/a[-1]) for x in a) if a != (0,) else a


def _inverse_mod(a, p, budget):
    """Return a unit's inverse; never assume a reducible quotient is a field."""
    old_r, r = p, a
    old_s, s = (F(0),), (F(1),)
    while r != (0,):
        q, remainder = divrem(old_r, r, budget)
        old_r, r = r, remainder
        old_s, s = s, sub(old_s, mul(q, s, budget), budget)
    if len(old_r) != 1 or not old_r[0]:
        raise ValueError('polynomial is not a unit modulo P')
    inverse = trim(tuple(_q(x/old_r[0]) for x in old_s), budget)
    return divrem(inverse, p, budget)[1]


def derivative(p, budget=None):
    p = trim(p, budget)
    return trim(tuple(_q(i*p[i]) for i in range(1, len(p))) or (0,), budget)


def eval(p, x, budget=None):
    p, x = trim(p, budget), _q(x)
    y = F(0)
    for c in reversed(p):
        _charge(budget)
        y = _q(y*x+c)
    return y


def _sturm(p, budget):
    seq = [p, derivative(p, budget)]
    while seq[-1] != (0,):
        r = divrem(seq[-2], seq[-1], budget)[1]
        if r == (0,):
            break
        seq.append(tuple(-x for x in r))
    return seq if seq[-1] != (0,) else seq[:-1]


def _variation(seq, x, budget):
    signs = [1 if v > 0 else -1 for p in seq if (v := eval(p, x, budget))]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _count(p, lo, hi, budget):
    s = _sturm(p, budget)
    return _variation(s, lo, budget)-_variation(s, hi, budget)


def _split(p, lo, hi, budget):
    # Avoid rational roots at the subdivision boundary without discarding them.
    for k in range(2, MAX_STEPS+2):
        mid = _q(lo+(hi-lo)/k)
        if eval(p, mid, budget):
            return mid
    raise BudgetExceeded('subdivision bound exceeded')


class Root:
    def __init__(self, poly, lo, hi, budget):
        self.budget = budget
        self.poly = trim(poly, budget)
        self.lo, self.hi = _q(lo), _q(hi)
        if not 2 <= len(self.poly) <= 4 or self.lo >= self.hi:
            raise ValueError('degree 1..3 and ordered open interval required')
        if gcd(self.poly, derivative(self.poly, budget), budget) != (1,):
            raise ValueError('square-free polynomial required')
        if not eval(self.poly, self.lo, budget) or not eval(self.poly, self.hi, budget):
            raise ValueError('root at interval endpoint')
        if _count(self.poly, self.lo, self.hi, budget) != 1:
            raise ValueError('interval must isolate exactly one root')

    def num(self, numerator, denominator=(1,)):
        return AN(self, numerator, denominator)

    def _zero(self, p):
        g = gcd(self.poly, p, self.budget)
        return len(g) > 1 and _count(g, self.lo, self.hi, self.budget) == 1

    def _refine(self):
        m = _split(self.poly, self.lo, self.hi, self.budget)
        if _count(self.poly, self.lo, m, self.budget):
            self.hi = m
        else:
            self.lo = m

    def to_data(self):
        return {'poly': _pdata(self.poly), 'lo': _qdata(self.lo), 'hi': _qdata(self.hi)}

    @classmethod
    def from_data(cls, data, budget):
        _fields(data, {'poly', 'lo', 'hi'})
        return cls(_pread(data['poly']), _qread(data['lo']), _qread(data['hi']), budget)


def isolate_real_roots(poly, budget):
    p = trim(poly, budget)
    if p == (0,):
        raise ValueError('zero polynomial has no finite root isolation')
    if len(p) == 1:
        return []
    if len(p) > 4:
        raise ValueError('degree bound is three')
    p = divrem(p, gcd(p, derivative(p, budget), budget), budget)[0]
    bound = _q(2+max(abs(x/p[-1]) for x in p[:-1]))
    todo, roots, steps = [(-bound, bound)], [], 0
    while todo:
        steps += 1
        if steps > MAX_STEPS:
            raise BudgetExceeded('isolation subdivision bound exceeded')
        lo, hi = todo.pop()
        n = _count(p, lo, hi, budget)
        if n == 1:
            roots.append(Root(p, lo, hi, budget))
        elif n:
            m = _split(p, lo, hi, budget)
            todo.extend([(m, hi), (lo, m)])
    return sorted(roots, key=lambda r: r.lo)


def _interval(p, lo, hi, budget):
    a = b = F(0)
    for c in reversed(p):
        _charge(budget, 4)
        values = [_q(a*lo), _q(a*hi), _q(b*lo), _q(b*hi)]
        a, b = _q(min(values)+c), _q(max(values)+c)
    return a, b


class AN:
    def __init__(self, root, numerator, denominator=(1,)):
        if type(root) is not Root:
            raise ValueError('Root required')
        self.root = root
        self.numerator = divrem(numerator, root.poly, root.budget)[1]
        self.denominator = divrem(denominator, root.poly, root.budget)[1]
        if root._zero(self.denominator):
            raise ZeroDivisionError('denominator vanishes at selected root')
        b = root.budget
        if self.numerator == (0,):
            self.denominator = (F(1),)
            return
        if self.denominator == (1,):
            return
        # A denominator coprime with P is a genuine unit even if P is
        # reducible. Otherwise retain a root-local rational function.
        if gcd(root.poly, self.denominator, b) == (1,):
            inverse = _inverse_mod(self.denominator, root.poly, b)
            self.numerator = divrem(mul(self.numerator, inverse, b), root.poly, b)[1]
            self.denominator = (F(1),)
        else:
            common = gcd(self.numerator, self.denominator, b)
            self.numerator = divrem(self.numerator, common, b)[0]
            self.denominator = divrem(self.denominator, common, b)[0]
            scale = self.denominator[-1]
            self.numerator = trim(tuple(_q(x/scale) for x in self.numerator), b)
            self.denominator = trim(tuple(_q(x/scale) for x in self.denominator), b)

    def _other(self, x):
        if type(x) is AN:
            if x.root is not self.root:
                raise ValueError('arithmetic requires the same Root object')
            return x
        return self.root.num((_q(x),))

    def __add__(self, other):
        o, b = self._other(other), self.root.budget
        return self.root.num(add(mul(self.numerator,o.denominator,b),mul(o.numerator,self.denominator,b),b), mul(self.denominator,o.denominator,b))
    __radd__ = __add__

    def __neg__(self):
        return self.root.num(tuple(-x for x in self.numerator),self.denominator)

    def __sub__(self, o):
        return self+-self._other(o)

    def __rsub__(self, o):
        return self._other(o)+-self

    def __mul__(self, other):
        o,b = self._other(other),self.root.budget
        return self.root.num(mul(self.numerator,o.numerator,b),mul(self.denominator,o.denominator,b))
    __rmul__ = __mul__

    def __truediv__(self, other):
        o,b = self._other(other),self.root.budget
        return self.root.num(mul(self.numerator,o.denominator,b),mul(self.denominator,o.numerator,b))

    def __rtruediv__(self, other):
        return self._other(other)/self

    def __pow__(self, exponent):
        if type(exponent) is not int or abs(exponent) > 4096:
            raise ValueError('bounded integer exponent required')
        if exponent < 0:
            return (1/self)**(-exponent)
        result, base = self.root.num((1,)), self
        while exponent:
            _charge(self.root.budget)
            if exponent & 1:
                result = result*base
            exponent //= 2
            if exponent:
                base = base*base
        return result

    def sign(self):
        r = self.root
        if r._zero(self.numerator):
            return 0
        for _ in range(MAX_STEPS):
            n,d = _interval(self.numerator,r.lo,r.hi,r.budget),_interval(self.denominator,r.lo,r.hi,r.budget)
            if (n[0]>0 or n[1]<0) and (d[0]>0 or d[1]<0):
                return (1 if n[0]>0 else -1)*(1 if d[0]>0 else -1)
            r._refine()
        raise BudgetExceeded('sign subdivision bound exceeded')

    def equals(self, other):
        return (self-self._other(other)).sign() == 0

    def __eq__(self, other):
        return self.equals(other)

    def __lt__(self, other):
        return (self-self._other(other)).sign()<0

    def __le__(self, other):
        return (self-self._other(other)).sign()<=0

    def __gt__(self, other):
        return (self-self._other(other)).sign()>0

    def __ge__(self, other):
        return (self-self._other(other)).sign()>=0

    def to_data(self):
        return {'numerator':_pdata(self.numerator),'denominator':_pdata(self.denominator)}

    @classmethod
    def from_data(cls, data, root):
        _fields(data, {'numerator','denominator'})
        return cls(root,_pread(data['numerator']),_pread(data['denominator']))


def _fields(data, fields):
    if type(data) is not dict or set(data) != fields:
        raise ValueError('exact JSON fields required')


def _qdata(q):
    return [q.numerator,q.denominator]


def _qread(data):
    if type(data) is not list or len(data)!=2 or any(type(x) is not int for x in data) or data[1]<=0:
        raise ValueError('rational JSON pair required')
    # Check before Fraction can normalize a malicious oversized pair.
    if max(abs(data[0]).bit_length(),data[1].bit_length())>MAX_BITS:
        raise BudgetExceeded('serialized rational bit bound exceeded')
    return _q(F(*data))


def _pdata(p):
    return [_qdata(x) for x in p]


def _pread(data):
    if type(data) is not list or not 1<=len(data)<=MAX_LENGTH:
        raise ValueError('bounded polynomial JSON list required')
    return tuple(_qread(x) for x in data)
