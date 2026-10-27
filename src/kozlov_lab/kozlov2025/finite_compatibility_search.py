"""Automatic finite sufficient-witness search for S-2025 PDF12.

Failure of either supported fit solver is never general incompatibility.
"""
from .given_m_compatibility import _group, given_m_compatibility, assess_given_m_compatibility_certificate
from .certified_granularity import granularity_from_witnesses, assess_granularity_certificate
from .cover_enumeration import _partitions
from .independent_cover import BudgetExceeded, verify_independent_cover_2d, verify_cover_certificate
from .primary_futlyar import assess_primary_futlyar, assess_primary_futlyar_certificate
from .zero_error_fits import _budget, _call
from .zero_error_fit_search import search_zero_error_fits
from .collinear_fits import collinear_fits
from .collinear_fits import assess_collinear_fits_certificate
from .four_center_fits import four_center_fits, assess_four_center_fits_certificate
from .circuit_fits import circuit_fits, assess_circuit_fits_certificate
from .zero_error_fits import assess_zero_error_fits_certificate
from .finite_environment import _canonical_rows

SCHEMA = 'finite-compatibility-search-v1'


def search_finite_compatibility(environment, group, *, max_work=2000000, seconds=30):
    """Descending m, all actual references, all full source label partitions.

    Stops on the first complete sufficient witness. m=1 is retained as an
    unsupported boundary because existing primary/fit contracts require m>=2.
    """
    b = None
    result = dict(schema=SCHEMA, status='UNKNOWN', certificate=None,
                  lower_bound=None, upper_bound=None, granularity=None,
                  general_compatibility_decided=False, visited_candidates=[],
                  counters=dict(reference_partitions=0, source_partitions=0,
                                zero_calls=0, collinear_calls=0, four_center_calls=0, circuit_calls=0, cache_hits=0))
    def finish(status, reason):
        result.update(status=status, reason=reason, work_used=b.used if b else 0)
        return result
    def call(fn, *args):
        if b.used >= b.maximum:
            raise BudgetExceeded('shared work budget exhausted before child call')
        return _call(b, fn, *args)
    try:
        b = _budget(max_work, seconds)
        raw, members = _group(environment, group, 0)
        upper = min(map(len, members))
        result['upper_bound'] = upper
        # At most Bell(7)=877; every generated partition is charged.
        partitions = {}
        for member in members:
            rows = []
            for index, parts in enumerate(_partitions(member)):
                b.spend()
                rows.append((index, tuple(tuple(p) for p in parts)))
            partitions[member] = rows
        fit_cache = {}
        for m in range(upper, 1, -1):
            for ri, member in enumerate(members):
                reference = {i: raw[i] for i in member}
                for pi, parts in partitions[member]:
                    if len(parts) != m:
                        continue
                    b.spend()
                    row = dict(kind='reference', id=(m, ri, pi), parts=parts,
                               status='PENDING')
                    result['visited_candidates'].append(row)
                    result['counters']['reference_partitions'] += 1
                    independent = call(verify_independent_cover_2d, reference, parts)
                    row['independence_witness'] = independent
                    if independent['status'] != 'VALID_INDEPENDENT_COVER':
                        row['status'] = independent['status']
                        continue
                    if not verify_cover_certificate(reference, parts, independent):
                        row['status'] = 'INVALID_LOCAL_WITNESS'
                        continue
                    primary = call(assess_primary_futlyar, reference, parts)
                    row['primary_result'] = primary
                    row['status'] = primary['status']
                    if primary['status'] != 'TRUE':
                        continue
                    pc = primary['certificate']
                    if call(assess_primary_futlyar_certificate, reference, pc)['status'] != 'VERIFIED':
                        row['status'] = 'UNVERIFIED_PRIMARY'
                        continue
                    fits = []
                    for si, source_member in enumerate(members):
                        source = {i: raw[i] for i in source_member}
                        found = None
                        for qi, source_parts in partitions[source_member]:
                            if len(source_parts) != m:
                                continue
                            b.spend()
                            candidate = dict(kind='source', id=(m, ri, pi, si, qi),
                                             parts=source_parts, attempts=[], status='PENDING')
                            result['visited_candidates'].append(candidate)
                            result['counters']['source_partitions'] += 1
                            key = (ri, parts, si, source_parts)
                            if key in fit_cache:
                                result['counters']['cache_hits'] += 1
                                found = fit_cache[key]
                                candidate['status'] = 'VERIFIED_CACHE'
                            else:
                                solvers = [
                                    (search_zero_error_fits, 'zero_calls', 'TRUE', assess_zero_error_fits_certificate),
                                    (collinear_fits, 'collinear_calls', 'COLLINEAR_FITS_SUFFICIENT_WITNESS', assess_collinear_fits_certificate)]
                                if m == 4:
                                    solvers.append((four_center_fits, 'four_center_calls',
                                        'FOUR_CENTER_FITS_SUFFICIENT_WITNESS', assess_four_center_fits_certificate))
                                elif 5 <= m <= 7:
                                    solvers.append((circuit_fits, 'circuit_calls',
                                        'CIRCUIT_FITS_SUFFICIENT_WITNESS', assess_circuit_fits_certificate))
                                for solver, counter, success, verifier in solvers:
                                    result['counters'][counter] += 1
                                    answer = call(solver, reference, parts, source, source_parts)
                                    candidate['attempts'].append(answer)
                                    if answer['status'] == success:
                                        cert = answer['certificate']
                                        check = call(verifier, reference, source, cert)
                                        candidate['attempts'].append(dict(kind='independent_check', **check))
                                        if check['status'] == 'VERIFIED':
                                            found = cert
                                            fit_cache[key] = found
                                            break
                                candidate['status'] = 'SUFFICIENT_WITNESS' if found else 'LOCAL_NO_WITNESS'
                            if found:
                                fits.append(found)
                                break
                        if found is None:
                            break
                    if len(fits) != len(members):
                        row['status'] = 'LOCAL_NO_ALL_MEMBER_WITNESS'
                        continue
                    witness = call(given_m_compatibility, raw, members, ri, parts, fits)
                    if witness['status'] != 'M_COMPATIBLE_SUFFICIENT_WITNESS':
                        row['status'] = witness['status']
                        continue
                    compatibility = witness['certificate']
                    if call(assess_given_m_compatibility_certificate, raw, compatibility)['status'] != 'VERIFIED':
                        row['status'] = 'UNVERIFIED_COMPATIBILITY'
                        continue
                    gc = None
                    if m == upper:
                        g = call(granularity_from_witnesses, raw, members, [compatibility])
                        if g['status'] != 'EXACT_GRANULARITY_CERTIFIED':
                            return finish('UNKNOWN', 'tight witness granularity verification incomplete')
                        gc = g['certificate']
                    result.update(lower_bound=m, granularity=m if gc else None,
                        certificate=dict(schema=SCHEMA, environment=tuple(sorted(raw.items())),
                            group=members, lower_bound=m, upper_bound=upper,
                            compatibility_certificate=compatibility, granularity_certificate=gc))
                    return finish('EXACT_GRANULARITY_CERTIFIED' if gc else 'LOWER_BOUND',
                                  'complete same-reference same-primary sufficient witness')
        return finish('UNKNOWN', 'supported finite candidates exhausted; m=1 and general fits unresolved')
    except BudgetExceeded as exc:
        for candidate in result['visited_candidates']:
            if candidate.get('status') == 'PENDING':
                candidate['status'] = 'UNKNOWN'
                candidate['reason'] = 'shared budget interrupted this candidate'
        return finish('UNKNOWN', str(exc))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        return finish('UNSUPPORTED_INPUT', str(exc))


def assess_finite_compatibility_certificate(environment, group, certificate,
                                           *, max_work=2000000, seconds=30):
    """Independent wrapper: existing verifiers and exact identities only."""
    b = None
    try:
        b = _budget(max_work, seconds)
        raw, members = _group(environment, group, 0)
        c = certificate
        if type(c) is not dict or set(c) != {'schema','environment','group','lower_bound','upper_bound','compatibility_certificate','granularity_certificate'}:
            raise ValueError('strict wrapper fields required')
        if (type(c['group']) is not tuple or any(type(member) is not tuple for member in c['group'])
                or _group(raw, c['group'], 0)[1] != members):
            raise ValueError('strict canonical group labels required')
        if c['schema'] != SCHEMA or not _canonical_rows(c['environment']) or c['environment'] != tuple(sorted(raw.items())) or c['group'] != members:
            raise ValueError('complete environment/group identity mismatch')
        upper = min(map(len, members))
        lower = c['lower_bound']
        if type(lower) is not int or type(c['upper_bound']) is not int or not 1 <= lower <= upper or c['upper_bound'] != upper:
            raise ValueError('strict exact bounds required')
        w = c['compatibility_certificate']
        if w['group'] != members or w['m'] != lower:
            raise ValueError('compatibility identity or bound mismatch')
        check = _call(b, assess_given_m_compatibility_certificate, raw, w)
        if check['status'] != 'VERIFIED':
            return dict(status=check['status'], work_used=b.used)
        gc = c['granularity_certificate']
        if lower == upper:
            if gc is None or gc['compatible_witness'] != w:
                raise ValueError('tight maximum requires matching granularity certificate')
            check = _call(b, assess_granularity_certificate, raw, gc)
            if check['status'] != 'VERIFIED':
                return dict(status=check['status'], work_used=b.used)
        elif gc is not None:
            raise ValueError('lower witness cannot claim maximum')
        return dict(status='VERIFIED', work_used=b.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN', reason=str(exc), work_used=b.used if b else 0)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        return dict(status='INVALID', reason=str(exc), work_used=b.used if b else 0)
