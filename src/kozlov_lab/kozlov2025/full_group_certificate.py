"""S-2025 PDF12 full groups: exact-M cardinality or given extension proof."""
from .given_m_compatibility import _group
from .certified_granularity import assess_granularity_certificate
from .finite_environment import _canonical_rows
from .zero_error_fits import _budget, _call
from .independent_cover import BudgetExceeded

SCHEMA = 'full-group-certificate-v1'


def _identity(raw, group, granularity):
    _, members = _group(raw, group, 0)
    if granularity['group'] != members or granularity['environment'] != tuple(sorted(raw.items())):
        raise ValueError('complete environment and group must match granularity witness')
    return members


def certify_full_group(environment, group, granularity_certificate, *,
                       extension_certificate=None, max_work=2000000, seconds=30):
    """Certify unused-label bound, or a supplied proper same-M extension.

    No general full/closed/etalon search; no member image is enlarged.
    """
    b = None
    def finish(status, reason, certificate=None):
        return dict(schema=SCHEMA, status=status, reason=reason, certificate=certificate,
                    work_used=b.used if b else 0)
    try:
        b = _budget(max_work, seconds)
        raw, members = _group(environment, group, 0)
        members = _identity(raw, members, granularity_certificate)
        check = _call(b, assess_granularity_certificate, raw, granularity_certificate)
        if check['status'] != 'VERIFIED':
            return finish('UNKNOWN', 'exact granularity unresolved or invalid')
        m = granularity_certificate['granularity']
        used = set().union(*map(set, members))
        unused = tuple(sorted(set(raw)-used))
        if extension_certificate is None:
            if len(unused) >= m:
                return finish('UNKNOWN', 'unused-label cardinality does not exclude same-M extensions')
            proof = 'unused_label_cardinality_strictly_less_than_exact_M'
            claim = 'FULL_GROUP_CERTIFIED'
        else:
            extension = _identity(raw, extension_certificate['group'], extension_certificate)
            check = _call(b, assess_granularity_certificate, raw, extension_certificate)
            if check['status'] != 'VERIFIED':
                return finish('UNKNOWN', 'given extension exact granularity unresolved or invalid')
            if not set(members) < set(extension) or extension_certificate['granularity'] != m:
                return finish('UNKNOWN', 'given extension must be proper and have the same exact M')
            proof = 'proper_image_set_extension_with_same_exact_M'
            claim = 'NOT_FULL_GIVEN_EXTENSION'
        c = dict(schema=SCHEMA, environment=tuple(sorted(raw.items())), group=members,
                 claim=claim, granularity=m, unused_labels=unused, proof=proof,
                 granularity_certificate=granularity_certificate,
                 extension_certificate=extension_certificate)
        return finish(claim, proof, c)
    except BudgetExceeded as exc:
        return finish('UNKNOWN', str(exc))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        return finish('UNSUPPORTED_INPUT', str(exc))


def assess_full_group_certificate(environment, group, certificate, *, max_work=2000000, seconds=30):
    """Only old granularity verifiers plus finite label/cardinality proofs."""
    b = None
    try:
        b = _budget(max_work, seconds)
        raw, members = _group(environment, group, 0)
        c = certificate
        if type(c) is not dict or set(c) != {'schema','environment','group','claim','granularity','unused_labels','proof','granularity_certificate','extension_certificate'}:
            raise ValueError('strict full-group certificate fields required')
        if (type(c['group']) is not tuple or any(type(member) is not tuple for member in c['group'])
                or _group(raw, c['group'], 0)[1] != members):
            raise ValueError('strict canonical group labels required')
        if c['schema'] != SCHEMA or not _canonical_rows(c['environment']) or c['environment'] != tuple(sorted(raw.items())) or c['group'] != members:
            raise ValueError('complete external environment/group identity mismatch')
        m = c['granularity']
        if type(m) is not int or m < 1:
            raise ValueError('strict exact M required')
        gc = c['granularity_certificate']
        _identity(raw, members, gc)
        check = _call(b, assess_granularity_certificate, raw, gc)
        if check['status'] != 'VERIFIED':
            return dict(status=check['status'],work_used=b.used)
        if gc['granularity'] != m:
            raise ValueError('exact M differs from verified granularity')
        unused = tuple(sorted(set(raw)-set().union(*map(set,members))))
        if type(c['unused_labels']) is not tuple or any(type(i) is not int for i in c['unused_labels']) or c['unused_labels'] != unused:
            raise ValueError('unused-label complement mismatch')
        ec = c['extension_certificate']
        if c['claim'] == 'FULL_GROUP_CERTIFIED':
            if ec is not None or c['proof'] != 'unused_label_cardinality_strictly_less_than_exact_M' or len(unused) >= m:
                raise ValueError('strict unused-cardinality proof missing')
        elif c['claim'] == 'NOT_FULL_GIVEN_EXTENSION':
            if ec is None or c['proof'] != 'proper_image_set_extension_with_same_exact_M':
                raise ValueError('proper extension proof missing')
            extension = _identity(raw, ec['group'], ec)
            check = _call(b, assess_granularity_certificate, raw, ec)
            if check['status'] != 'VERIFIED':
                return dict(status=check['status'],work_used=b.used)
            if not set(members) < set(extension) or ec['granularity'] != m:
                raise ValueError('extension is not proper same-M group')
        else:
            raise ValueError('unsupported full-group claim')
        b.spend(0)
        return dict(status='VERIFIED',work_used=b.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=b.used if b else 0)
