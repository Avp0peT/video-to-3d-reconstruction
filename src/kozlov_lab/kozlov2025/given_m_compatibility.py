"""S-2025 PDF12: one group member, one primary futlyar, all members.

Given sufficient witnesses only; no granularity or existential compatibility search.
"""
from .affine_subimage import _rows
from .zero_error_fits import _budget, _call, _parts, assess_zero_error_fits_certificate
from .primary_futlyar import assess_primary_futlyar_certificate
from .collinear_fits import assess_collinear_fits_certificate
from .four_center_fits import assess_four_center_fits_certificate
from .circuit_fits import assess_circuit_fits_certificate
from .independent_cover import BudgetExceeded

SCHEMA = 'given-m-compatibility-v1'


def _group(environment, group, reference_index):
    raw = dict(_rows(environment, 7))
    if type(group) not in (list, tuple) or not group:
        raise ValueError('nonempty explicit separate group required; empty-group boundary excluded')
    seen, members = set(), []
    for member in group:
        if type(member) not in (list, tuple) or not member or any(type(i) is not int for i in member):
            raise ValueError('nonempty labelled members required')
        ids = set(member)
        if len(ids)!=len(member) or ids & seen or not ids <= raw.keys():
            raise ValueError('member identities foreign, duplicate or intersecting')
        seen |= ids
        members.append(tuple(sorted(ids)))
    if type(reference_index) is not int or not 0<=reference_index<len(members):
        raise ValueError('reference must be a member of the group')
    return raw, tuple(members)


def _f_object(pc):
    # Proof representations may differ; these independently certified fields
    # determine the same mathematical A, its given cover, centres and fixed R.
    return (pc['source_coordinate_sha256'],tuple(pc['declared_labels']),
            tuple(tuple(x) for x in pc['parts']),tuple(tuple(x) for x in pc['centers']),pc['radius_squared'])


def given_m_compatibility(environment, group, reference_index, reference_parts,
                          member_certificates, *, max_work=2000000, seconds=30):
    """All member certificates must use the same specified group member and F."""
    try:
        raw, members = _group(environment, group, reference_index)
        reference = {i:raw[i] for i in members[reference_index]}
        parts = _parts(reference, reference_parts)
        if type(member_certificates) not in (tuple,list) or len(member_certificates)!=len(members):
            raise ValueError('exactly one certificate per group member required')
        certificate = dict(schema=SCHEMA, environment=tuple(sorted(raw.items())),group=members,
            reference_index=reference_index, reference_parts=parts,m=len(parts),
            shared_primary=member_certificates[0]['primary_certificate'],
            member_certificates=tuple(member_certificates))
        check = assess_given_m_compatibility_certificate(environment,certificate,max_work=max_work,seconds=seconds)
        return dict(schema=SCHEMA,status='M_COMPATIBLE_SUFFICIENT_WITNESS' if check['status']=='VERIFIED'
                    else 'UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID_WITNESS',
                    reason=check['reason'],work_used=check['work_used'],
                    certificate=certificate if check['status']=='VERIFIED' else None,
                    general_compatibility_decided=False)
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        return dict(schema=SCHEMA,status='INVALID_WITNESS',reason=str(exc),work_used=0,
                    certificate=None,general_compatibility_decided=False)


def assess_given_m_compatibility_certificate(environment,certificate,*,max_work=2000000,seconds=30):
    """Independent complete quantifier/witness checks; invokes no producer/search."""
    budget=None
    def finish(status,reason):
        return dict(status=status,reason=reason,work_used=budget.used if budget else 0)
    try:
        budget=_budget(max_work,seconds)
        c=certificate
        if type(c) is not dict or c['schema']!=SCHEMA:
            return finish('INVALID','certificate schema invalid')
        raw,members=_group(environment,c['group'],c['reference_index'])
        from .finite_environment import _canonical_rows
        if not _canonical_rows(c['environment']) or c['environment']!=tuple(sorted(raw.items())) or c['group']!=members:
            return finish('INVALID','complete environment/group binding mismatch')
        reference={i:raw[i] for i in members[c['reference_index']]}
        parts=_parts(reference,c['reference_parts'])
        if type(c['m']) is not int or c['m']!=len(parts) or type(c['member_certificates']) is not tuple or len(c['member_certificates'])!=len(members):
            return finish('INVALID','m or all-member quantifier incomplete')
        primary=_call(budget,assess_primary_futlyar_certificate,reference,c['shared_primary'])
        if primary['status']!='VERIFIED':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'INVALID','shared primary not independently verified')
        if tuple(tuple(p) for p in c['shared_primary']['parts'])!=parts:
            return finish('INVALID','specified reference cover differs from shared primary')
        common=_f_object(c['shared_primary'])
        for member,fit in zip(members,c['member_certificates']):
            source={i:raw[i] for i in member}
            if fit['reference_points']!=tuple(sorted(reference.items())) or fit['reference_parts']!=parts or len(fit['source_parts'])!=c['m']:
                return finish('INVALID','member witness uses another reference or m')
            if fit.get('schema')=='zero-error-fits-v1':
                verifier=assess_zero_error_fits_certificate
            elif fit.get('schema')=='collinear-fits-v1':
                verifier=assess_collinear_fits_certificate
            elif fit.get('schema')=='four-center-fits-v1':
                verifier=assess_four_center_fits_certificate
            elif fit.get('schema')=='circuit-fits-v1':
                verifier=assess_circuit_fits_certificate
            else:
                return finish('INVALID','unsupported sufficient fits certificate schema')
            check=_call(budget,verifier,reference,source,fit)
            if check['status']!='VERIFIED':
                return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','member complete fits witness invalid')
            if _f_object(fit['primary_certificate'])!=common:
                return finish('INVALID','member witness uses a different mathematical primary futlyar')
        budget.spend(0)
        return finish('VERIFIED','one reference member and one primary futlyar contain every given group member')
    except BudgetExceeded as exc:
        return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return finish('INVALID',str(exc))
