"""Bounded centres of all independent covers of a complete supplied 1..7 set.

Reuse the existing enumeration and exact constructors. This is a finite given
environment, not general ostov, futlyar, fits or semantic discovery.
"""
import math
import time
import hashlib
import json
from fractions import Fraction

from .cover_enumeration import enumerate_independent_covers_2d, _partitions, _bell
from .independent_cover import normalize, verify_independent_cover_2d, verify_cover_certificate
from .ostov_supported_blocks import given_cover_supported_block_ostov, verify_supported_block_ostov
from .exact_contact3_ellipse import complete_input_contact3_ellipse, verify_complete_input_contact3_ellipse
from .exact_contact5_ellipse import complete_input_contact5_ellipse, verify_complete_input_contact5_ellipse
from .exact_contact4_ellipse import complete_input_contact4_ellipse
from .exact_contact4_verify import verify_complete_input_contact4_ellipse
from .exact_triangle_ellipse import _digest


def _enumeration_digest(raw):
    rows = [[hex(i), [[str(v.numerator), str(v.denominator)] for v in raw[i]]] for i in sorted(raw)]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def build_cover_center_family(points, *, max_candidates=877, total_max_work=2000000, seconds=30):
    """Enumerate once; preserve every certified cover including UNKNOWN rows.

    All child work is charged to one decreasing ledger and all child times use
    one deadline. Cached blocks share an immutable mathematical input identity.
    Rejected partition certificates are retained for solver-free verification
    of completeness; their additional construction is charged too.
    """
    start = time.monotonic()
    result = dict(schema='cover-center-family-v2', status='UNSUPPORTED_INPUT', complete=False,
                  enumeration_complete=False, decision_ledger_complete=False,
                  families=[], decision_ledger=[], work_used=0,
                  scope='finite complete supplied 1..7 environment; not general theory')
    if (type(max_candidates) is not int or not 0 <= max_candidates <= 877 or
            type(total_max_work) is not int or not 0 <= total_max_work <= 2000000 or
            type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return dict(result, reason='invalid bounded shared budget')
    deadline = start + seconds
    result['budgets'] = dict(max_candidates=max_candidates, total_max_work=total_max_work, seconds=seconds)
    def finish(status, reason):
        result.update(status=status, reason=reason, elapsed_seconds=time.monotonic()-start)
        result['exact_covers'] = sum(r['status'] == 'EXACT' for r in result['families'])
        result['unknown_covers'] = len(result['families'])-result['exact_covers']
        return result
    def remaining():
        work = total_max_work-result['work_used']
        duration = deadline-time.monotonic()
        if work <= 0 or duration <= 0:
            raise TimeoutError('shared work/time exhausted')
        return work, duration
    def charge(answer):
        result['work_used'] += answer.get('work_used', 0)
        if result['work_used'] > total_max_work or time.monotonic() >= deadline:
            raise TimeoutError('shared work/time exhausted')
    try:
        if not isinstance(points, dict) or not 1 <= len(points) <= 7:
            return finish('UNSUPPORTED_INPUT', 'complete input must have 1..7 distinct labelled points')
        raw, _, _ = normalize(points, [sorted(points)])
        result.update(declared_labels=sorted(raw), source_coordinate_sha256=_digest(raw),
                      expected_candidates=_bell(len(raw)))
        work, duration = remaining()
        enumeration = enumerate_independent_covers_2d(raw, max_candidates=max_candidates,
            total_max_work=work, seconds=duration)
        result['enumeration'] = enumeration
        result['enumeration_complete'] = enumeration['complete']
        # Register every enumerated cover before any later operation can stop.
        for row in enumeration['independent_covers']:
            result['families'].append(dict(candidate_index=row['candidate_index'], parts=row['parts'],
                cover_certificate=row['certificate'], status='UNKNOWN', reason='not processed', blocks=[]))
        charge(enumeration)
        accepted = {r['candidate_index']: r for r in result['families']}
        # The existing enumeration discards rejected certificates. Reconstruct
        # only those witnesses to make the completeness claim independently checkable.
        for index, parts in enumerate(_partitions(sorted(raw))):
            if index >= enumeration['judged_candidates']:
                break
            work, duration = remaining()
            if index in accepted:
                certificate = accepted[index]['cover_certificate']
            else:
                certificate = verify_independent_cover_2d(raw, parts, max_work=work, seconds=duration)
                charge(certificate)
            if certificate['status'] not in ('VALID_INDEPENDENT_COVER', 'NOT_INDEPENDENT') or not verify_cover_certificate(raw, parts, certificate):
                return finish('PARTIAL_UNKNOWN', 'decision ledger witness unavailable')
            result['decision_ledger'].append(dict(candidate_index=index, parts=parts, certificate=certificate))
        result['decision_ledger_complete'] = len(result['decision_ledger']) == result['expected_candidates']
        cache = {}
        for row in result['families']:
            block_rows = []
            for group in row['parts']:
                work, duration = remaining()
                key = tuple(group)
                if key not in cache:
                    block = {i: raw[i] for i in group}
                    answer = given_cover_supported_block_ostov(block, [group], max_work=work, seconds=duration)
                    charge(answer)
                    if answer['status'] == 'EXACT_GIVEN_COVER_OSTOV':
                        cache[key] = dict(labels=list(group), kind='supported', certificate=answer,
                            center=answer['centers'][0], size=answer['pieces'][0]['size'])
                    elif answer['status'] == 'UNSUPPORTED_DOMAIN':
                        work, duration = remaining()
                        answer = complete_input_contact3_ellipse(block, max_work=work, seconds=duration)
                        charge(answer)
                        if answer['status'] == 'EXACT_OPTIMUM_COMPLETE_INPUT':
                            cache[key] = dict(labels=list(group), kind='contact3', certificate=answer['certificate'],
                                center=answer['center'], size=answer['certificate']['major_axis_length'])
                        elif answer['status'] == 'UNSUPPORTED_CONTACT_DOMAIN' and len(group) >= 4:
                            if len(group) >= 5:
                                work, duration = remaining()
                                answer = complete_input_contact5_ellipse(block, max_work=work, seconds=duration)
                                charge(answer)
                            if answer['status'] == 'EXACT_OPTIMUM_COMPLETE_INPUT':
                                cache[key] = dict(labels=list(group), kind='contact5', certificate=answer['certificate'],
                                    center=answer['center'], size=answer['certificate']['major_axis_length'])
                            elif answer['status'] == 'UNSUPPORTED_CONTACT_DOMAIN':
                                work, duration = remaining()
                                answer = complete_input_contact4_ellipse(block, max_work=work, seconds=duration)
                                charge(answer)
                                if answer['status'] == 'EXACT_OPTIMUM_COMPLETE_INPUT':
                                    z = answer['certificate']
                                    cache[key] = dict(labels=list(group), kind='contact4', certificate=z,
                                        center=dict(expression='algebraic_vector', root=z['algebraic_root'], coordinates=z['center']),
                                        size=dict(expression='algebraic_length', root=z['algebraic_root'], value=z['major_axis_length']))
                                else:
                                    cache[key] = dict(labels=list(group), kind='UNKNOWN', reason=answer['status'])
                            else:
                                cache[key] = dict(labels=list(group), kind='UNKNOWN', reason=answer['status'])
                        else:
                            cache[key] = dict(labels=list(group), kind='UNKNOWN', reason=answer['status'])
                    else:
                        cache[key] = dict(labels=list(group), kind='UNKNOWN', reason=answer['status'])
                block_rows.append(cache[key])
            row['blocks'] = block_rows
            if all(b['kind'] != 'UNKNOWN' for b in block_rows):
                row.update(status='EXACT', reason='all complete blocks independently certifiable',
                           centers=[b['center'] for b in block_rows],
                           cover_size=dict(expression='max(nonnegative_lengths)', lengths=[b['size'] for b in block_rows]))
            else:
                row['reason'] = 'unsupported exact contact domain; cover retained'
        remaining()
        result['complete'] = (result['enumeration_complete'] and result['decision_ledger_complete'] and
                              all(r['status'] == 'EXACT' for r in result['families']))
        return finish('COMPLETE' if result['complete'] else 'PARTIAL_UNKNOWN',
                      'all finite covers exact' if result['complete'] else 'unresolved covers or enumeration')
    except TimeoutError as exc:
        return finish('PARTIAL_UNKNOWN', str(exc))
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as exc:
        return finish('UNSUPPORTED_INPUT', str(exc))


def verify_cover_center_family(points, result):
    """Solver-free check of block witnesses and exhaustive partition decisions.

    True certifies the declared partial/complete result, not completion alone.
    No hull, centre construction, cover solver or enumeration solver is called.
    """
    try:
        def exact_tree(value):
            if isinstance(value, dict):
                return all(exact_tree(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(exact_tree(v) for v in value)
            return type(value) in (int, Fraction, str)
        if not isinstance(points, dict) or not 1 <= len(points) <= 7:
            return False
        raw, _, _ = normalize(points, [sorted(points)])
        r = result
        if (r['schema'] not in ('cover-center-family-v1', 'cover-center-family-v2') or any(type(i) is not int for i in r['declared_labels']) or r['declared_labels'] != sorted(raw) or
                r['source_coordinate_sha256'] != _digest(raw) or r['expected_candidates'] != _bell(len(raw))):
            return False
        e = r['enumeration']
        if (type(e['complete']) is not bool or r['enumeration_complete'] is not e['complete'] or
                any(type(i) is not int for i in e['declared_labels']) or e['declared_labels'] != sorted(raw) or
                e['source_coordinate_sha256'] != _enumeration_digest(raw) or
                e['expected_candidates'] != r['expected_candidates']):
            return False
        partitions = list(_partitions(sorted(raw)))
        if (type(e['judged_candidates']) is not int or not 0 <= e['judged_candidates'] <= len(partitions) or
                type(e['visited_candidates']) is not int or not e['judged_candidates'] <= e['visited_candidates'] <= len(partitions)):
            return False
        if e['complete'] and (e['status'] != 'COMPLETE' or e['judged_candidates'] != len(partitions) or e['uncertain_candidates']):
            return False
        covers = e['independent_covers']
        indices = [c['candidate_index'] for c in covers]
        if (any(type(i) is not int or not 0 <= i < e['judged_candidates'] for i in indices) or
                indices != sorted(set(indices)) or len(covers)+e['rejected_candidates'] != e['judged_candidates']):
            return False
        for c in covers:
            if (c['parts'] != partitions[c['candidate_index']] or c['source_coordinate_sha256'] != e['source_coordinate_sha256'] or
                    c['certificate']['status'] != 'VALID_INDEPENDENT_COVER' or
                    not verify_cover_certificate(raw, c['parts'], c['certificate'])):
                return False
        ledger = r['decision_ledger']
        if len(ledger) > e['judged_candidates']:
            return False
        for i, decision in enumerate(ledger):
            if (type(decision['candidate_index']) is not int or decision['candidate_index'] != i or
                    decision['parts'] != partitions[i] or not verify_cover_certificate(raw, partitions[i], decision['certificate'])):
                return False
            status = decision['certificate']['status']
            if status not in ('VALID_INDEPENDENT_COVER', 'NOT_INDEPENDENT') or (status == 'VALID_INDEPENDENT_COVER') != (i in indices):
                return False
        ledger_complete = len(ledger) == len(partitions)
        if r['decision_ledger_complete'] is not ledger_complete:
            return False
        rows = r['families']
        if len(rows) != len(covers):
            return False
        for c, row in zip(covers, rows):
            _, _, missing = normalize(raw, row['parts'])
            if missing:
                return False
            if (type(row['candidate_index']) is not int or row['candidate_index'] != c['candidate_index'] or row['parts'] != c['parts'] or
                    row['cover_certificate'] != c['certificate'] or row['status'] not in ('EXACT', 'UNKNOWN')):
                return False
            blocks = row['blocks']
            if len(blocks) > len(row['parts']):
                return False
            for group, block_row in zip(row['parts'], blocks):
                if any(type(i) is not int for i in block_row['labels']) or block_row['labels'] != group:
                    return False
                block = {i: raw[i] for i in group}
                kind = block_row['kind']
                if kind == 'UNKNOWN':
                    continue
                z = block_row['certificate']
                if kind == 'supported':
                    if not verify_supported_block_ostov(block, z) or z['parts'] != [group]:
                        return False
                    center, size = z['centers'][0], z['pieces'][0]['size']
                elif kind == 'contact3':
                    if not verify_complete_input_contact3_ellipse(block, z):
                        return False
                    center, size = z['center'], z['major_axis_length']
                elif kind == 'contact5':
                    if not verify_complete_input_contact5_ellipse(block, z):
                        return False
                    center, size = z['center'], z['major_axis_length']
                elif kind == 'contact4' and r['schema'] == 'cover-center-family-v2':
                    if not verify_complete_input_contact4_ellipse(block, z):
                        return False
                    center = dict(expression='algebraic_vector', root=z['algebraic_root'], coordinates=z['center'])
                    size = dict(expression='algebraic_length', root=z['algebraic_root'], value=z['major_axis_length'])
                else:
                    return False
                if (not exact_tree(block_row['center']) or not exact_tree(block_row['size']) or
                        block_row['center'] != center or block_row['size'] != size):
                    return False
            if row['status'] == 'EXACT':
                if (not exact_tree(row['centers']) or not exact_tree(row['cover_size']) or
                        len(blocks) != len(row['parts']) or any(b['kind'] == 'UNKNOWN' for b in blocks) or
                        row['centers'] != [b['center'] for b in blocks] or
                        row['cover_size'] != dict(expression='max(nonnegative_lengths)', lengths=[b['size'] for b in blocks])):
                    return False
        exact = sum(row['status'] == 'EXACT' for row in rows)
        complete = e['complete'] and ledger_complete and exact == len(rows)
        return (r['complete'] is complete and r['exact_covers'] == exact and r['unknown_covers'] == len(rows)-exact and
                r['status'] == ('COMPLETE' if complete else 'PARTIAL_UNKNOWN'))
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError):
        return False
