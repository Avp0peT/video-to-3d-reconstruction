"""Exact supplied-point research UI, isolated from photo/CAD execution."""
from pathlib import Path
import sys
import hashlib
import json
import time
from fractions import Fraction
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
from scripts import verify_generalized_futlyar as cli
from kozlov_lab.kozlov2025.cover_center_family import build_cover_center_family, verify_cover_center_family
from scripts.solve_exact_center import input_points as center_input_points
from kozlov_lab.kozlov2025.exact_center import solve_exact_center, assess_exact_center_result, encode_exact, decode_exact
from kozlov_lab.kozlov2025.independent_cover import Budget, BudgetExceeded
from kozlov_lab.kozlov2025.affine_subimage import find_affine_subimage_2d, verify_affine_subimage_certificate, AffineSubimageCertificate
from kozlov_lab.kozlov2025.finite_environment import (enumerate_environment_parts, verify_environment_parts,
    enumerate_separate_groups, verify_separate_groups_certificate, EnvironmentParts, SeparateGroupsCertificate)
from kozlov_lab.kozlov2025.algebraic_primary_futlyar import assess_algebraic_primary_futlyar, assess_algebraic_primary_certificate
from kozlov_lab.kozlov2025.zero_error_fit_search import search_zero_error_fits, assess_zero_error_fit_search_result
from kozlov_lab.kozlov2025.collinear_fits import collinear_fits, assess_collinear_fits_certificate


# Same lossless tagged serializer as RUN20261013_cover_center_family/run_once.py.
# That run script has top-level execution, so importing it would execute work.
def exact_encode(x):
    if isinstance(x, Fraction): return {'fraction': [str(x.numerator), str(x.denominator)]}
    if isinstance(x, dict): return {'mapping': [[exact_encode(k), exact_encode(v)] for k, v in x.items()]}
    if isinstance(x, tuple): return {'tuple': [exact_encode(v) for v in x]}
    if isinstance(x, list): return [exact_encode(v) for v in x]
    return x


def exact_decode(x):
    if isinstance(x, dict):
        if set(x) == {'fraction'}: return Fraction(*map(int, x['fraction']))
        if set(x) == {'mapping'}: return {exact_decode(k): exact_decode(v) for k, v in x['mapping']}
        if set(x) == {'tuple'}: return tuple(exact_decode(v) for v in x['tuple'])
        raise ValueError('unknown exact encoding')
    if isinstance(x, list): return [exact_decode(v) for v in x]
    return x


def parse(text):
    if len(text.encode('utf-8')) > 4 << 20: raise ValueError('JSON exceeds 4 MiB')
    value = json.loads(text, object_pairs_hook=cli.unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(value) is not dict: raise ValueError('JSON object required')
    return value


def compute_family(text, *, max_work=2000000, seconds=10):
    data = parse(text)
    if set(data) != {'schema', 'points'} or data['schema'] != 'cover-center-family-input-v1':
        raise ValueError('expected cover-center-family-input-v1 with complete points')
    points = cli.points(data['points'])
    if not 1 <= len(points) <= 7: raise ValueError('完整输入须为 1–7 个点；不会抽样')
    answer = build_cover_center_family(points, max_candidates=877, total_max_work=max_work, seconds=seconds)
    payload = json.dumps(exact_encode({'input': points, 'result': answer,
        'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'scope': 'complete supplied exact points only; independent mathematical branch',
        'physical_scale': 'unknown', 'photo_quality': 'not evaluated'}), ensure_ascii=False, allow_nan=False)
    restored = exact_decode(json.loads(payload))
    checked = verify_cover_center_family(restored['input'], restored['result'])
    return {'status': answer['status'], 'independent_check': checked, 'complete': answer['complete'],
            'unknown_covers': answer.get('unknown_covers', 0), 'family_count': len(answer['families']),
            'payload': payload, 'witness': payload}


def compute_futlyar(text, *, max_work=262144):
    data = parse(text)
    p, c, r = cli.input_values(data)
    answer = cli.verify_generalized_futlyar_2d(p, c, r, max_work=max_work)
    result = cli.encode(asdict(answer))
    checked = 'NOT_AVAILABLE'
    if answer.certificate is not None:
        restored = cli.certificate_from_json(json.loads(json.dumps(result['certificate'])))
        checked = cli.verify_generalized_futlyar_certificate(p, c, r, restored, max_work=max_work)
    result.update(schema='generalized-futlyar-result-v1', input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  independent_check=checked, physical_scale='unknown', photo_quality='not evaluated', scope='supplied exact placement only')
    return {'status': answer.status, 'independent_check': checked, 'reason': answer.reason,
            'payload': json.dumps(result, ensure_ascii=False, allow_nan=False),
            'witness': json.dumps({'certificate': result['certificate'], 'witness': result['witness']}, ensure_ascii=False, allow_nan=False)}


def _new_points(rows, maximum=7):
    points = cli.points(rows)
    if not 1 <= len(points) <= maximum: raise ValueError(f'完整输入须为 1–{maximum} 个点；不会抽样')
    return points


def _proof_pack(value, *, split_ledgers=False):
    """Explicit independently bounded exact pieces, preserving container identity."""
    if split_ledgers and type(value) is dict:
        return dict(kind='mapping',entries=[[encode_exact(k),_proof_pack(v,split_ledgers=True)] for k,v in value.items()])
    if split_ledgers and type(value) is tuple and len(value)>512:
        return dict(kind='tuple',items=[_proof_pack(v) for v in value])
    encoded=encode_exact(value)
    try:
        decode_exact(encoded)
        return dict(kind='exact',value=encoded)
    except ValueError:
        if type(value) is dict:
            return dict(kind='mapping',entries=[[encode_exact(k),_proof_pack(v)] for k,v in value.items()])
        if type(value) in (tuple,list):
            return dict(kind='tuple' if type(value) is tuple else 'list',items=[_proof_pack(v) for v in value])
        raise


def _proof_unpack(document):
    remaining=[20000]
    def restore(piece,depth=0):
        remaining[0]-=1
        if remaining[0]<0 or depth>32 or type(piece) is not dict:
            raise ValueError('bounded explicit proof pieces required')
        if piece.get('kind')=='exact' and set(piece)=={'kind','value'}:
            return decode_exact(piece['value'])
        if piece.get('kind') in ('tuple','list') and set(piece)=={'kind','items'}:
            if type(piece['items']) is not list or len(piece['items'])>20000:
                raise ValueError('bounded proof container required')
            items=[restore(v,depth+1) for v in piece['items']]
            return tuple(items) if piece['kind']=='tuple' else items
        if piece.get('kind')=='mapping' and set(piece)=={'kind','entries'}:
            if type(piece['entries']) is not list or len(piece['entries'])>20000:
                raise ValueError('bounded proof entries required')
            result={}
            for row in piece['entries']:
                if type(row) is not list or len(row)!=2: raise ValueError('proof mapping entry shape')
                key=decode_exact(row[0])
                if type(key) not in (str,int) or key in result: raise ValueError('unique exact proof keys required')
                result[key]=restore(row[1],depth+1)
            return result
        raise ValueError('invalid explicit proof piece')
    return restore(document)


def _new_payload(text, inputs, result, scope):
    # New modes explicitly use exact_center $exact tags. Legacy family mapping/
    # tuple/fraction tags stay unchanged; the envelope declares the format.
    envelope = dict(schema='exact-research-tagged-v1',
        encoding='exact-center-tags-v1', input=inputs, result=result,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(), scope=scope,
        physical_scale='unknown', photo_quality='not evaluated')
    # S+ at seven labels has 4139 records. Preserve the decoder's per-tree
    # refusal gate by carrying explicit bounded group chunks, not relaxing it.
    if 'environment_parts' in result and len(result.get('groups',()))>512:
        base=dict(envelope); base['result']=dict(result)
        groups=base['result'].pop('groups')
        if result['certificate'] is not None:
            base['result']['certificate']=dict(result['certificate'])
            if base['result']['certificate'].pop('groups')!=groups: raise ValueError('environment group binding mismatch')
        transport=dict(schema='exact-research-chunked-v1',encoding='exact-center-tags-v1',document=encode_exact(base),
            group_chunks=[encode_exact(dict(offset=i,groups=groups[i:i+128])) for i in range(0,len(groups),128)])
    elif (result.get('schema')=='circuit-fits-v1' and
          len(result.get('certificate',{}).get('alignment_certificate',{}).get('ledger',()))>512):
        # Full six/seven-centre ledgers use the existing lossless proof pieces.
        transport=dict(schema='exact-research-proof-chunked-v1',encoding='exact-center-tags-v1',document=_proof_pack(envelope,split_ledgers=True))
    else: transport=encode_exact(envelope)
    try:
        restored=decode_new_payload(transport)
    except ValueError:
        if ('visited_candidates' not in result and result.get('schema')!='circuit-fits-v1'
                and result.get('status') not in ('EXACT_ENVIRONMENT_CODE','EXACT_RESTRICTED_CODE','EXACT_ELLIPSE_CODE',
                'PROPORTIONAL_GIVEN_MAPPING','NOT_PROPORTIONAL_GIVEN_MAPPING')): raise
        transport=dict(schema='exact-research-proof-chunked-v1',encoding='exact-center-tags-v1',document=_proof_pack(envelope))
        restored=decode_new_payload(transport)
    payload=json.dumps(transport,ensure_ascii=False,allow_nan=False)
    return payload, decode_new_payload(json.loads(payload))


def decode_new_payload(value, *, max_work=2000000, seconds=30):
    """New tagged envelope, with explicit S+ chunks for large finite groups."""
    if type(value) is dict and value.get('schema')=='exact-research-ellipse-equivalence2019-v1':
        from kozlov_lab.coding.ellipse_equivalence2019 import decode_ellipse_equivalence2019
        if (set(value)!={'schema','encoding','document','certificate'}
                or value['encoding']!='ellipse-equivalence-2019-json-v1'):
            raise ValueError('invalid ellipse equivalence transport')
        document=decode_exact(value['document'])
        if (type(document) is not dict or document.get('schema')!='exact-research-tagged-v1'
                or type(document.get('result')) is not dict or 'certificate' in document['result']):
            raise ValueError('invalid ellipse equivalence metadata')
        document['result']['certificate']=(decode_ellipse_equivalence2019(value['certificate'],max_work=max_work,seconds=seconds)
                                           if value['certificate'] is not None else None)
        return document
    if type(value) is dict and value.get('schema')=='exact-research-proof-chunked-v1':
        if set(value)!={'schema','encoding','document'} or value['encoding']!='exact-center-tags-v1':
            raise ValueError('invalid explicit proof transport')
        return _proof_unpack(value['document'])
    if type(value) is dict and value.get('schema')=='exact-research-chunked-v1':
        if set(value)!={'schema','encoding','document','group_chunks'} or value['encoding']!='exact-center-tags-v1':
            raise ValueError('invalid chunked transport')
        chunks=value['group_chunks']
        if type(chunks) is not list or len(chunks)>40: raise ValueError('bounded environment chunks required')
        envelope=decode_exact(value['document']); groups=()
        for encoded in chunks:
            chunk=decode_exact(encoded)
            if set(chunk)!={'offset','groups'} or type(chunk['offset']) is not int or chunk['offset']!=len(groups):
                raise ValueError('ordered environment chunk offsets required')
            if type(chunk['groups']) is not tuple or not 1<=len(chunk['groups'])<=128: raise ValueError('bounded group chunk required')
            groups+=chunk['groups']
        if len(groups)>4140: raise ValueError('environment record bound')
        envelope['result']['groups']=groups
        if envelope['result']['certificate'] is not None: envelope['result']['certificate']['groups']=groups
        return envelope
    return decode_exact(value)


def _view(answer, checked, payload, **summary):
    return dict(status=answer['status'], independent_check=checked, reason=answer.get('reason',''),
                complete=answer.get('complete',False), payload=payload, witness=payload, **summary)


def compute_center(text, *, max_work=2000000, seconds=10):
    points = center_input_points(parse(text))
    started = time.monotonic()
    answer = solve_exact_center(points, max_work=max_work, seconds=seconds)
    payload, restored = _new_payload(text, points, answer, '完整给定点集的精确中心和长轴；有限精确域')
    checked = 'NOT_AVAILABLE'
    if answer['exact_optimum']:
        checked = assess_exact_center_result(restored['input'], restored['result'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer, checked, payload, center=answer.get('center'), major_axis_length=answer.get('major_axis_length'),
                 profile=answer.get('profile','未完成'))


def compute_affine(text, *, max_work=2000000, seconds=10):
    data = parse(text)
    if set(data)!={'schema','source','target'} or data['schema']!='affine-subimage-input-v1':
        raise ValueError('expected affine-subimage-input-v1 with complete source/target')
    source,target = _new_points(data['source'],12),_new_points(data['target'],32)
    budget = Budget(max_work,seconds)
    answer = asdict(find_affine_subimage_2d(source,target,budget=budget))
    payload, restored = _new_payload(text,dict(source=source,target=target),answer,'完整 source 仿射映射到 target 的子像；不验证 fits')
    checked = 'NOT_AVAILABLE'
    if answer['certificate'] is not None:
        cert = AffineSubimageCertificate(**restored['result']['certificate'])
        checked = verify_affine_subimage_certificate(restored['input']['source'],restored['input']['target'],cert,budget=budget)
    return _view(answer,checked,payload)


def compute_environment(text, *, max_work=2000000, seconds=10):
    data = parse(text)
    if set(data)!={'schema','points','include_empty'} or data['schema']!='finite-environment-input-v1' or type(data['include_empty']) is not bool:
        raise ValueError('expected finite-environment-input-v1 with complete points and bool include_empty')
    points = _new_points(data['points']); budget=Budget(max_work,seconds)
    try:
        budget.spend(1+(1<<len(points)))
        parts = enumerate_environment_parts(points)
        groups = enumerate_separate_groups(points,include_empty=data['include_empty'],budget=budget)
        answer = asdict(groups); answer['environment_parts']=asdict(parts)
    except BudgetExceeded as exc:
        answer = dict(status='PARTIAL_UNKNOWN',complete=False,reason=str(exc),groups=(),certificate=None,
                      work_used=budget.used,environment_parts=None)
    payload, restored = _new_payload(text,points,answer,'有限环境非空部分与标签互斥 S+；不计算几何独立性或 fits；S* 仅基数')
    checked = 'NOT_AVAILABLE'; parts_checked = False
    if answer['environment_parts'] is not None:
        parts_checked = verify_environment_parts(restored['input'],EnvironmentParts(**restored['result']['environment_parts']))
    if answer['certificate'] is not None:
        checked=verify_separate_groups_certificate(restored['input'],SeparateGroupsCertificate(**restored['result']['certificate']),budget=budget)
    return _view(answer,checked,payload,parts_count=len(answer['environment_parts']['parts']) if answer['environment_parts'] else 0,
                 group_count=len(answer['groups']),parts_check=parts_checked,
                 star_cardinality=answer['environment_parts']['star_cardinality'] if answer['environment_parts'] else None)


def compute_primary(text, *, max_work=2000000, seconds=10):
    data=parse(text)
    if set(data)!={'schema','points','parts'} or data['schema']!='primary-futlyar-input-v1':
        raise ValueError('expected primary-futlyar-input-v1 with complete points/parts')
    points=_new_points(data['points']); started=time.monotonic()
    answer=assess_algebraic_primary_futlyar(points,data['parts'],max_work=max_work,seconds=seconds)
    payload,restored=_new_payload(text,dict(points=points,parts=data['parts']),answer,'仅给定覆盖的 primary；不搜索覆盖或通用 fits')
    checked='NOT_AVAILABLE'
    if answer['status']=='TRUE' and answer.get('certificate') is not None:
        checked=assess_algebraic_primary_certificate(restored['input']['points'],restored['result']['certificate'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer,checked,payload,check_note='FALSE 保留算法反例；独立入口仅核验 TRUE 证书，未另核验反例。' if answer['status']=='FALSE' else '')


def compute_fit_search(text, *, max_work=2000000, seconds=10):
    data=parse(text)
    if set(data)!={'schema','reference','reference_parts','source','source_parts'} or data['schema']!='zero-error-fit-search-input-v1':
        raise ValueError('expected zero-error-fit-search-input-v1 with complete reference/source and parts')
    reference,source=_new_points(data['reference']),_new_points(data['source'])
    started=time.monotonic()
    answer=search_zero_error_fits(reference,data['reference_parts'],source,data['source_parts'],max_work=max_work,seconds=seconds)
    payload,restored=_new_payload(text,dict(reference=reference,source=source,reference_parts=data['reference_parts'],source_parts=data['source_parts']),
        answer,'有限零中心误差 fits 充分证据搜索；不判定通用 fits 不存在')
    checked='NOT_AVAILABLE'
    if answer['status']=='TRUE':
        checked=assess_zero_error_fit_search_result(restored['input']['reference'],restored['input']['source'],restored['result'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer,checked,payload,center_rank=answer.get('center_rank'),candidates_tried=answer.get('candidates_tried',0),
                 general_fits_decided=answer['general_fits_decided'],
                 check_note='未找到本有限搜索支持的见证；不证明通用 fits 不存在。秩 1 横向扩展仍受限。' if answer['status']=='NO_SUPPORTED_WITNESS' else '')


def compute_collinear_fits(text, *, max_work=2000000, seconds=10):
    data=parse(text)
    if set(data)!={'schema','reference','reference_parts','source','source_parts'} or data['schema']!='collinear-fits-input-v1':
        raise ValueError('expected collinear-fits-input-v1 with complete reference/source and parts')
    reference,source=_new_points(data['reference']),_new_points(data['source'])
    started=time.monotonic()
    answer=collinear_fits(reference,data['reference_parts'],source,data['source_parts'],max_work=max_work,seconds=seconds)
    payload,restored=_new_payload(text,dict(reference=reference,source=source,reference_parts=data['reference_parts'],source_parts=data['source_parts']),
        answer,'共线中心、给定分划、固定横向延拓的严格 fits 充分证据；不判定通用 fits')
    checked='NOT_AVAILABLE';error=None
    if answer['status']=='COLLINEAR_FITS_SUFFICIENT_WITNESS':
        error=answer['certificate']['alignment_certificate']['error_squared']
        checked=assess_collinear_fits_certificate(restored['input']['reference'],restored['input']['source'],restored['result']['certificate'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer,checked,payload,error_squared=error,general_fits_decided=answer['general_fits_decided'],
        check_note='固定横向延拓未获得完整严格包含见证；不证明通用 fits 不存在。' if answer['status']=='NO_SUPPORTED_WITNESS' else '')


def compute_four_center_fits(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.kozlov2025.four_center_fits import four_center_fits,assess_four_center_fits_certificate
    data=parse(text)
    if set(data)!={'schema','reference','reference_parts','source','source_parts'} or data['schema']!='four-center-fits-input-v1':
        raise ValueError('expected four-center-fits-input-v1 with complete reference/source and parts')
    reference,source=_new_points(data['reference']),_new_points(data['source'])
    started=time.monotonic()
    answer=four_center_fits(reference,data['reference_parts'],source,data['source_parts'],max_work=max_work,seconds=seconds)
    payload,restored=_new_payload(text,dict(reference=reference,source=source,reference_parts=data['reference_parts'],source_parts=data['source_parts']),
        answer,'四中心非共线全局稳定互位及完整块严格 fits 充分证据；不判定通用 fits')
    checked='NOT_AVAILABLE';error=None
    if answer['status']=='FOUR_CENTER_FITS_SUFFICIENT_WITNESS':
        error=answer['certificate']['alignment_certificate']['error_squared']
        checked=assess_four_center_fits_certificate(restored['input']['reference'],restored['input']['source'],restored['result']['certificate'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer,checked,payload,error_squared=error,general_fits_decided=answer['general_fits_decided'],
        check_note='仅返回四中心全局稳定与严格包含的充分证据；未支持或未知不证明通用 fits 不存在。')


def compute_compatibility(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.kozlov2025.finite_compatibility_search import search_finite_compatibility,assess_finite_compatibility_certificate
    from kozlov_lab.kozlov2025.full_group_certificate import certify_full_group,assess_full_group_certificate
    data=parse(text)
    if set(data)!={'schema','environment','group'} or data['schema']!='finite-compatibility-search-input-v1':
        raise ValueError('expected finite-compatibility-search-input-v1 with complete environment and label group')
    environment=_new_points(data['environment']);group=data['group'];started=time.monotonic();used=0
    def call(function,*args):
        nonlocal used
        answer=function(*args,max_work=max(0,max_work-used),seconds=max(0,seconds-(time.monotonic()-started)))
        used+=answer['work_used'];return answer
    answer=call(search_finite_compatibility,environment,group)
    checked='NOT_AVAILABLE';full=dict(status='NOT_AVAILABLE',certificate=None,work_used=0);full_check='NOT_AVAILABLE'
    if answer.get('certificate') is not None:
        # Read back the full search result, including its candidate ledger, before verification.
        _,restored=_new_payload(text,dict(environment=environment,group=group),answer,'自动有限群兼容充分证据搜索；未找到不表示一般不兼容')
        checked=call(assess_finite_compatibility_certificate,environment,group,restored['result']['certificate'])['status']
        if checked=='VERIFIED' and answer['status']=='EXACT_GRANULARITY_CERTIFIED':
            full=call(certify_full_group,environment,group,restored['result']['certificate']['granularity_certificate'])
            if full.get('certificate') is not None:
                # Exact certificate roundtrip is mandatory before the independent full-group check.
                _,full_readback=_new_payload(text,dict(environment=environment,group=group),full,'基于精确粒度与未用标签数的完整群充分证明')
                full_check=call(assess_full_group_certificate,environment,group,full_readback['result']['certificate'])['status']
    result=dict(answer);result.update(full_group_result=full,compatibility_independent_check=checked,
        full_group_independent_check=full_check,total_work_used=used)
    payload,restored=_new_payload(text,dict(environment=environment,group=group),result,'自动群兼容/精确粒度/有限完整群充分证据；完整环境保留；不判定一般不兼容或 closed/etalon')
    return _view(result,checked,payload,lower_bound=answer['lower_bound'],upper_bound=answer['upper_bound'],
        granularity=answer.get('granularity'),full_group_status=full['status'],full_group_check=full_check,
        candidates_tried=len(answer['visited_candidates']),general_compatibility_decided=answer['general_compatibility_decided'],
        check_note='LOWER_BOUND 仅下界；UNKNOWN 或未找到支持见证不表示一般不兼容。完整群只在精确粒度与未用标签界成立时证明。')


def compute_circuit_fits(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.kozlov2025.circuit_fits import circuit_fits, assess_circuit_fits_certificate
    data=parse(text)
    if set(data)!={'schema','reference','reference_parts','source','source_parts'} or data['schema']!='circuit-fits-input-v1':
        raise ValueError('expected circuit-fits-input-v1 with complete reference/source and parts')
    reference,source=_new_points(data['reference']),_new_points(data['source'])
    started=time.monotonic()
    answer=circuit_fits(reference,data['reference_parts'],source,data['source_parts'],max_work=max_work,seconds=seconds)
    payload,restored=_new_payload(text,dict(reference=reference,source=source,reference_parts=data['reference_parts'],source_parts=data['source_parts']),
        answer,'完整4–7有理中心的四点关系下界与全块严格包含充分证书；不判定一般 fits')
    checked='NOT_AVAILABLE';error=None
    if answer['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS':
        error=answer['certificate']['alignment_certificate']['error_squared']
        checked=assess_circuit_fits_certificate(restored['input']['reference'],restored['input']['source'],restored['result']['certificate'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))['status']
    return _view(answer,checked,payload,error_squared=error,general_fits_decided=answer['general_fits_decided'],
        check_note='四点关系仅给出多中心充分证书；UNKNOWN 或未支持不证明通用 fits 不存在。所选变换未通过全块包含时，不声称尝试过所有稳定变换。')


def _registration_input(text, schema, extra):
    data=parse(text)
    if set(data)!={'schema','source','target'}|set(extra) or data['schema']!=schema:
        raise ValueError('expected '+schema+' with complete source/target and explicit fields')
    return data,_new_points(data['source'],32),_new_points(data['target'],32)


def _registration_view(text, inputs, answer, assessor, started, max_work, seconds, scope, **summary):
    payload,restored=_new_payload(text,inputs,answer,scope)
    checked='NOT_AVAILABLE';check_work=0
    if answer.get('certificate') is not None:
        readback=assessor(restored['input'],restored['result']['certificate'],
            max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))
        checked=readback['status'];check_work=readback['work_used']
    view=_view(answer,checked,payload,**summary)
    view['complete']=answer['status'].startswith('EXACT_')
    view['total_work_used']=answer['work_used']+check_work
    view['check_note']=scope+'；算法状态与独立回读核验分别展示。'
    return view


def compute_translation(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.registration.fixed_orientation import solve_fixed_orientation_translation, assess_fixed_orientation_certificate
    data,source,target=_registration_input(text,'fixed-orientation-translation-input-v1',('mapping','orientation'))
    rows=data['mapping'];mapping={}
    if type(rows) is not list or not 1<=len(rows)<=32: raise ValueError('explicit bounded mapping rows required')
    for row in rows:
        if (type(row) is not dict or set(row)!={'source','target'} or
            type(row['source']) is not int or type(row['target']) is not int or row['source'] in mapping):
            raise ValueError('unique integer source/target correspondence required')
        mapping[row['source']]=row['target']
    q=data['orientation']
    if type(q) is not list or len(q)!=2 or any(type(row) is not list or len(row)!=2 for row in q):
        raise ValueError('explicit 2x2 rational orthogonal orientation required')
    q=tuple(tuple(cli.rational(v) for v in row) for row in q)
    inputs=dict(source=source,target=target,mapping=mapping,orientation=q)
    started=time.monotonic()
    answer=solve_fixed_orientation_translation(source,target,mapping,orientation=q,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        return assess_fixed_orientation_certificate(i['source'],i['target'],i['mapping'],c,orientation=i['orientation'],**budget)
    cert=answer.get('certificate') or {}
    return _registration_view(text,inputs,answer,assessor,started,max_work,seconds,
        '完整1–32点，固定有理正交朝向与显式给定对应的精确最优平移；可反射；不搜索全局转角或对应',
        translation=cert.get('translation'),distance_squared=cert.get('radius_squared'))


def compute_candidate_acceptance(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.registration.candidate_matching import candidate_acceptance_exact, assess_candidate_acceptance_certificate
    data,source,target=_registration_input(text,'placed-candidate-input-v1',('radius_squared',))
    radius=cli.rational(data['radius_squared']);started=time.monotonic()
    answer=candidate_acceptance_exact(source,target,radius,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        if c.get('radius_squared')!=i['radius_squared']: return dict(status='INVALID',work_used=0)
        return assess_candidate_acceptance_certificate(i['source'],i['target'],c,**budget)
    cert=answer.get('certificate') or {}
    return _registration_view(text,dict(source=source,target=target,radius_squared=radius),answer,assessor,started,max_work,seconds,
        '完整1–32点的给定位置与半径平方接受性；普通接受只需两侧邻居，窄接受需双射；Hall缺陷独立核验；不搜索位置',
        acceptable=answer.get('acceptable'),narrow_acceptable=answer.get('narrow_acceptable'),obstruction=cert.get('obstruction'))


def compute_placed_bottleneck(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.registration.candidate_matching import placed_bottleneck_exact, assess_placed_bottleneck_certificate
    data,source,target=_registration_input(text,'placed-bottleneck-input-v1',())
    started=time.monotonic()
    answer=placed_bottleneck_exact(source,target,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        return assess_placed_bottleneck_certificate(i['source'],i['target'],c,**budget)
    return _registration_view(text,dict(source=source,target=target),answer,assessor,started,max_work,seconds,
        '完整等长1–32点在给定位置的精确瓶颈匹配；阈值最小性证书；不搜索旋转、平移或全局配准',
        distance_squared=answer.get('distance_squared'))


def compute_environment_code(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.coding.environment_code import restrict_environment_code, assess_restricted_code
    data=parse(text)
    if set(data)!={'schema','environment','subset'} or data['schema']!='environment-code-input-v1':
        raise ValueError('expected environment-code-input-v1 with complete environment/subset')
    points=_new_points(data['environment'],8)
    labels=data['subset']
    if type(labels) is not list or not labels or any(type(i) is not int for i in labels):
        raise ValueError('nonempty integer global subset label array required')
    started=time.monotonic()
    answer=restrict_environment_code(points,labels,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        return assess_restricted_code(i['environment'],i['subset'],c,**budget)
    cert=answer.get('certificate') or {};env=cert.get('environment',{});sub=cert.get('restricted',{})
    return _registration_view(text,dict(environment=points,subset=tuple(labels)),answer,assessor,started,max_work,seconds,
        'S-2019-CODES pp57–58，完整1–8点环境面积比代码按全局标签限制；保留退化零分子与零分母账本；不判仿射等价',
        global_labels=env.get('labels',tuple(sorted(points))),subset_labels=sub.get('labels',tuple(labels)),
        environment_ratio_count=len(env.get('ratios',{})),ratio_count=len(sub.get('ratios',{})),
        triangle_count=len(sub.get('double_areas',{})),zero_denominator_count=len(sub.get('zero_denominators',())),
        plane_domain=sub.get('plane_domain'),environment_plane_domain=env.get('plane_domain'),code_defined=sub.get('code_defined'))


def compute_ellipse_code2019(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.coding.ellipse_code2019 import build_ellipse_code2019, assess_ellipse_code2019
    data=parse(text)
    if set(data)!={'schema','points'} or data['schema']!='ellipse-code2019-input-v1':
        raise ValueError('expected ellipse-code2019-input-v1 with complete points')
    points=_new_points(data['points'],8)
    if len(points)<3:raise ValueError('完整椭圆代码输入须为3–8点；不会抽样')
    started=time.monotonic()
    answer=build_ellipse_code2019(points,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        return assess_ellipse_code2019(i['points'],c,**budget)
    cert=answer.get('certificate') or {};area=cert.get('area_code',{})
    return _registration_view(text,dict(points=points),answer,assessor,started,max_work,seconds,
        'S-2019-CODES p59，完整3–8点且每个三元组非共线；精确三角形最小外接椭圆与面积比；不判仿射等价',
        global_labels=area.get('labels',tuple(sorted(points))),ratio_count=len(cert.get('ellipse_ratios',{})),
        ellipse_count=len(cert.get('ellipses',{})),plane_domain=area.get('plane_domain'))


def compute_proportionality(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.coding.environment_proportionality import paper_proportionality_given_mapping, assess_proportionality_certificate
    data=parse(text)
    fields={'schema','env','source_labels','target_labels','mapping'}
    if set(data)!=fields or data['schema']!='environment-proportionality-input-v1':
        raise ValueError('expected environment-proportionality-input-v1 with complete env and explicit labels/mapping')
    env=_new_points(data['env'],8)
    source,target=data['source_labels'],data['target_labels']
    for labels in (source,target):
        if type(labels) is not list or not 1<=len(labels)<=8 or any(type(i) is not int for i in labels):
            raise ValueError('nonempty bounded integer global part labels required')
    rows=data['mapping'];mapping={};used_targets=set()
    if type(rows) is not list or not 1<=len(rows)<=8:raise ValueError('explicit bounded mapping rows required')
    for row in rows:
        if (type(row) is not dict or set(row)!={'source','target'} or
                type(row['source']) is not int or type(row['target']) is not int or
                row['source'] in mapping or row['target'] in used_targets):
            raise ValueError('unique integer source/target correspondence required')
        mapping[row['source']]=row['target'];used_targets.add(row['target'])
    inputs=dict(env=env,source_labels=tuple(source),target_labels=tuple(target),mapping=mapping)
    started=time.monotonic()
    answer=paper_proportionality_given_mapping(env,source,target,mapping,max_work=max_work,seconds=seconds)
    def assessor(i,c,**budget):
        return assess_proportionality_certificate(i['env'],i['source_labels'],i['target_labels'],i['mapping'],c,**budget)
    cert=answer.get('certificate') or {}
    view=_registration_view(text,inputs,answer,assessor,started,max_work,seconds,
        'S-2019-CODES p58，完整环境内显式给定双射的单向面积比例定义；各部分可重叠；正负仅针对给定对应，不判未知对应或非仿射',
        paper_proportionality_given_mapping=answer.get('paper_proportionality_given_mapping'),
        degeneracy_preserved=answer.get('degeneracy_preserved'),full_area_proportionality=answer.get('full_area_proportionality'),
        theorem_eligible=answer.get('theorem_eligible'),constant=cert.get('constant'),
        global_labels=tuple(sorted(env)),source_labels=tuple(source),target_labels=tuple(target),
        ledger_count=len(cert.get('ledger',())),self_pair_count=sum(row['environment_ratio_reference']=='SELF_PAIR_EXCLUDED' for row in cert.get('ledger',())))
    view['complete']=answer['status'] in ('PROPORTIONAL_GIVEN_MAPPING','NOT_PROPORTIONAL_GIVEN_MAPPING')
    return view


def compute_ellipse_equivalence2019(text, *, max_work=2000000, seconds=10):
    from kozlov_lab.coding.ellipse_equivalence2019 import search_ellipse_equivalence2019, encode_ellipse_equivalence2019
    from kozlov_lab.registration.fixed_orientation import _Exhausted
    data=parse(text)
    if set(data)!={'schema','source','target'} or data['schema']!='ellipse-equivalence2019-input-v1':
        raise ValueError('expected ellipse-equivalence2019-input-v1 with complete source/target')
    source,target=_new_points(data['source'],8),_new_points(data['target'],8)
    if min(len(source),len(target))<3:raise ValueError('完整两侧须各3–8点；不会抽样')
    started=time.monotonic()
    answer=search_ellipse_equivalence2019(source,target,max_work=max_work,seconds=seconds)
    encoded_certificate=None
    if answer.get('certificate') is not None:
        try:
            encoded_certificate=encode_ellipse_equivalence2019(answer['certificate'],
                max_work=max(0,max_work-answer['work_used']),seconds=max(0,seconds-(time.monotonic()-started)))
        except _Exhausted:
            answer=dict(status='UNKNOWN',reason='专用证书序列化预算耗尽；不发布部分证书',certificate=None,
                work_used=answer['work_used'],code_equivalent=None,theorem_conclusion='NOT_DECIDED',affine_mapping_status='NOT_ESTABLISHED')
    certificate=answer.get('certificate') or {}
    metadata=dict(schema='exact-research-tagged-v1',encoding='ellipse-equivalence-2019-json-v1',
        input=dict(source=source,target=target),result={k:v for k,v in answer.items() if k!='certificate'},
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='S-2019-CODES pp58–59，完整3–8有理点、每个三元组非共线的未知编号双射搜索；不恢复相机、三维或CAD',
        physical_scale='unknown',photo_quality='not evaluated')
    transport=dict(schema='exact-research-ellipse-equivalence2019-v1',encoding='ellipse-equivalence-2019-json-v1',
                   document=encode_exact(metadata),certificate=encoded_certificate)
    payload=json.dumps(transport,ensure_ascii=False,allow_nan=False)
    return dict(status=answer['status'],independent_check='NOT_RUN' if certificate else 'NOT_AVAILABLE',
        reason=answer.get('reason',''),complete=answer['status'] in ('ELLIPSE_EQUIVALENT','NOT_ELLIPSE_EQUIVALENT'),
        payload=payload,witness=payload,has_certificate=bool(certificate),
        code_equivalent=answer.get('code_equivalent'),theorem_domain=answer.get('theorem_domain'),
        theorem_conclusion=answer.get('theorem_conclusion','NOT_DECIDED'),
        affine_mapping_status=answer.get('affine_mapping_status','NOT_ESTABLISHED'),
        ratio_count=len(certificate.get('ratio_ledger',())),
        rejected_correspondences=sum(len(chunk) for chunk in certificate.get('rejection_chunks',())),
        check_note='构造含算法内部核验；请另点独立核验按钮，对下载载荷无损回读后的证书验算。UNKNOWN不证明不存在对应。')


def verify_ellipse_equivalence2019_payload(payload, *, max_work=2000000, seconds=10):
    from kozlov_lab.coding.ellipse_equivalence2019 import assess_ellipse_equivalence2019
    from kozlov_lab.registration.fixed_orientation import _Budget, _Exhausted
    # Validate budgets even for a document without a mathematical certificate.
    _Budget(max_work,seconds)
    if len(payload.encode('utf-8'))>64<<20:raise ValueError('ellipse proof JSON exceeds 64 MiB')
    value=json.loads(payload,object_pairs_hook=cli.unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(value) is not dict or value.get('schema')!='exact-research-ellipse-equivalence2019-v1':
        raise ValueError('ellipse equivalence download required')
    started=time.monotonic()
    try:
        # Separate action budget: reserve half for decoding, half for independent
        # mathematics; the codec deliberately does not report mathematical work.
        restored=decode_new_payload(value,max_work=max_work//2,seconds=seconds)
        certificate=restored['result']['certificate']
        if certificate is None:
            return dict(status='NOT_AVAILABLE',reason='没有证书；UNKNOWN或UNSUPPORTED不能独立确认为正负结论')
        if restored['result'].get('status')!=certificate.get('verdict'):
            return dict(status='INVALID',reason='下载算法状态与证书结论不一致')
        checked=assess_ellipse_equivalence2019(restored['input']['source'],restored['input']['target'],certificate,
            max_work=max_work-max_work//2,seconds=max(0,seconds-(time.monotonic()-started)))
        if checked['status']=='VERIFIED':
            for key in ('code_equivalent','theorem_domain','theorem_conclusion','affine_mapping_status'):
                actual=restored['result'].get(key);expected=checked[key]
                if type(actual) is not type(expected) or actual!=expected:
                    return dict(status='INVALID',reason='下载摘要与独立核验不一致：'+key)
        return checked
    except _Exhausted:
        return dict(status='UNKNOWN',reason='专用证书回读预算耗尽；未完成独立核验')


ELLIPSE_EQUIVALENCE2019_EXAMPLE=json.dumps(dict(schema='ellipse-equivalence2019-input-v1',
    source=[dict(label=i,xy=p) for i,p in zip((11,22,33,44),((0,0),(2,0),(0,2),(3,3)))],
    target=[dict(label=i,xy=p) for i,p in zip((101,102,103,104),((16,2),(9,2),(11,-6),(7,-4)))]),indent=2)


PROPORTIONALITY_EXAMPLE=json.dumps(dict(schema='environment-proportionality-input-v1',
    env=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(1,0),(2,0),(0,1),(1,1)))],
    source_labels=[0,1,2,3],target_labels=[0,4,2,3],
    mapping=[dict(source=i,target=j) for i,j in ((0,0),(1,4),(2,2),(3,3))]),indent=2)


ENVIRONMENT_CODE_EXAMPLE=json.dumps(dict(schema='environment-code-input-v1',
    environment=[dict(label=i,xy=p) for i,p in zip((11,22,33,44,55),((0,0),(2,0),(0,2),(3,3),(5,1)))],
    subset=[11,22,33,44]),indent=2)
ELLIPSE_CODE2019_EXAMPLE=json.dumps(dict(schema='ellipse-code2019-input-v1',
    points=[dict(label=i,xy=p) for i,p in zip((11,22,33,44),((0,0),(2,0),(0,2),(3,3)))]),indent=2)


TRANSLATION_EXAMPLE=json.dumps(dict(schema='fixed-orientation-translation-input-v1',
    source=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(10,0),(0,10)))],
    target=[dict(label=i+10,xy=p) for i,p in enumerate(((0,0),(6,0),(-2,7)))],
    mapping=[dict(source=i,target=i+10) for i in range(3)],orientation=[[1,0],[0,1]]),indent=2)
CANDIDATE_ACCEPTANCE_EXAMPLE=json.dumps(dict(schema='placed-candidate-input-v1',
    source=[dict(label=i,xy=[x,0]) for i,x in enumerate((0,1,10))],
    target=[dict(label=i+10,xy=[x,0]) for i,x in enumerate((0,9,10))],radius_squared=1),indent=2)
PLACED_BOTTLENECK_EXAMPLE=json.dumps(dict(schema='placed-bottleneck-input-v1',
    source=json.loads(CANDIDATE_ACCEPTANCE_EXAMPLE)['source'],
    target=json.loads(CANDIDATE_ACCEPTANCE_EXAMPLE)['target']),indent=2)


FAMILY_EXAMPLE = json.dumps({'schema': 'cover-center-family-input-v1',
    'points': [{'label': i, 'xy': xy} for i, xy in enumerate([[0, 0], [1, 0], [1, 1], [0, 1]])]}, indent=2)
FUTLYAR_EXAMPLE = json.dumps({'schema': 'generalized-futlyar-input-v1',
    'points': [{'label': 7, 'xy': [0, 0]}, {'label': 9, 'xy': [{'numerator': '1', 'denominator': '2'}, 0]}, {'label': 11, 'xy': [3, 0]}],
    'centers': [{'label': 20, 'xy': [0, 0]}, {'label': 30, 'xy': [3, 0]}], 'radius': 1}, indent=2)
CENTER_EXAMPLE = json.dumps(dict(schema='exact-center-input-v1',points=[dict(label=i,xy=p) for i,p in enumerate(
    [[0,0],[1,0],[0,1],[2,dict(numerator='3',denominator='2')]])]),indent=2)
AFFINE_EXAMPLE = json.dumps(dict(schema='affine-subimage-input-v1',source=[dict(label=i,xy=p) for i,p in enumerate([[0,0],[1,0],[0,1]])],
    target=[dict(label=i+10,xy=p) for i,p in enumerate([[3,4],[5,4],[3,7],[20,20]])]),indent=2)
ENVIRONMENT_EXAMPLE = json.dumps(dict(schema='finite-environment-input-v1',points=[dict(label=i,xy=p) for i,p in enumerate([[0,0],[1,0],[0,1]])],include_empty=False),indent=2)
PRIMARY_EXAMPLE = json.dumps(dict(schema='primary-futlyar-input-v1',points=[dict(label=0,xy=[0,0]),dict(label=1,xy=[3,4])],parts=[[0],[1]]),indent=2)
PRIMARY_CUBIC_EXAMPLE = json.dumps(dict(schema='primary-futlyar-input-v1',
    points=json.loads(CENTER_EXAMPLE)['points']+[dict(label=4,xy=[100,100])],parts=[[0,1,2,3],[4]]),indent=2)
FIT_SEARCH_EXAMPLE = json.dumps(dict(schema='zero-error-fit-search-input-v1',
    reference=[dict(label=i,xy=p) for i,p in enumerate([[-1,-1],[1,-1],[0,2],[3,0],[0,100]])],reference_parts=[[0,1,2],[3],[4]],
    source=[dict(label=i,xy=[dict(numerator=str(n),denominator=str(d)) for n,d in xy]) for i,xy in enumerate([
        [(-3,2),(-2,3)],[(-1,2),(-2,3)],[(-1,1),(1,3)],[(1,2),(-1,3)],[(-1,1),(33,1)]])],
    source_parts=[[4],[0,1,2],[3]]),indent=2)
COLLINEAR_FITS_EXAMPLE = json.dumps(dict(schema='collinear-fits-input-v1',
    reference=[dict(label=i,xy=[x,0]) for i,x in enumerate([0,1,3])],reference_parts=[[0],[1],[2]],
    source=[dict(label=i+10,xy=[x,0]) for i,x in enumerate([0,1,2])],source_parts=[[10],[11],[12]]),indent=2)


FOUR_CENTER_FITS_EXAMPLE=json.dumps(dict(schema='four-center-fits-input-v1',
    reference=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(3,0),(0,2),(4,2)))],reference_parts=[[0],[1],[2],[3]],
    source=[dict(label=i+10,xy=p) for i,p in enumerate(((0,0),(1,0),(0,1),(1,1)))],source_parts=[[10],[11],[12],[13]]),indent=2)
COMPATIBILITY_EXAMPLE=json.dumps(dict(schema='finite-compatibility-search-input-v1',
    environment=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(1,0),(3,0),(0,10),(1,10),(2,10),(99,99)))],
    group=[[0,1,2],[3,4,5]]),indent=2)
CIRCUIT_FITS_EXAMPLE=json.dumps(dict(schema='circuit-fits-input-v1',
    reference=[dict(label=i,xy=p) for i,p in enumerate((
        (dict(numerator='-1',denominator='10'),0),(dict(numerator='101',denominator='10'),0),
        (dict(numerator='1',denominator='10'),20),(dict(numerator='99',denominator='10'),20),(31,47)))],
    reference_parts=[[i] for i in range(5)],
    source=[dict(label=i+10,xy=p) for i,p in enumerate(((0,0),(10,0),(0,20),(10,20),(31,47)))],
    source_parts=[[i+10] for i in range(5)]),indent=2)


def render():
    import streamlit as st
    st.title('精确点集研究（独立数学侧支）')
    st.warning('仅完整人工输入的二维数学验证。照片质量、几何产线与物理尺度均未评估；数学 TRUE 不表示 CAD 完成。')
    st.caption('默认内容是人工示例。坐标只接受整数或约分后的 numerator/denominator 对象；不读取照片，不抽样。')
    modes = {'完整点集覆盖中心族':('family',FAMILY_EXAMPLE,compute_family),
             '给定中心与半径 Futlyar 验证':('futlyar',FUTLYAR_EXAMPLE,compute_futlyar),
             '单个完整点集精确中心':('center',CENTER_EXAMPLE,compute_center),
             '精确仿射子像':('affine',AFFINE_EXAMPLE,compute_affine),
             '有限环境与标签互斥 S+':('environment',ENVIRONMENT_EXAMPLE,compute_environment),
             '给定覆盖 primary 验证':('primary',PRIMARY_EXAMPLE,compute_primary),
             '零中心误差 fits 搜索':('fit_search',FIT_SEARCH_EXAMPLE,compute_fit_search),
             '共线中心严格 fits':('collinear_fits',COLLINEAR_FITS_EXAMPLE,compute_collinear_fits),
             '四中心非共线严格 fits':('four_center_fits',FOUR_CENTER_FITS_EXAMPLE,compute_four_center_fits),
             '自动群兼容、粒度与完整群':('compatibility',COMPATIBILITY_EXAMPLE,compute_compatibility),
             '多中心下界证书 fits':('circuit_fits',CIRCUIT_FITS_EXAMPLE,compute_circuit_fits),
             '固定朝向与给定对应最优平移':('translation',TRANSLATION_EXAMPLE,compute_translation),
             '给定位置接受性':('candidate_acceptance',CANDIDATE_ACCEPTANCE_EXAMPLE,compute_candidate_acceptance),
             '给定位置瓶颈匹配':('placed_bottleneck',PLACED_BOTTLENECK_EXAMPLE,compute_placed_bottleneck),
             '环境代码与子图像限制':('environment_code',ENVIRONMENT_CODE_EXAMPLE,compute_environment_code),
             '2019椭圆代码':('ellipse_code2019',ELLIPSE_CODE2019_EXAMPLE,compute_ellipse_code2019),
             '环境内给定对应比例性':('proportionality',PROPORTIONALITY_EXAMPLE,compute_proportionality),
             '2019椭圆代码未知对应等价':('ellipse_equivalence2019',ELLIPSE_EQUIVALENCE2019_EXAMPLE,compute_ellipse_equivalence2019)}
    mode = st.selectbox('研究功能', list(modes), key='exact_mode')
    name, example, compute = modes[mode]
    family = mode == '完整点集覆盖中心族'
    if name=='environment': st.caption('S+ 仅按标签互斥；不代表凸包互斥或兼容。S* 只给出基数，不枚举。')
    if name=='primary': st.caption('仅核验给定覆盖；完整最多 7 点、最多一个四接触代数根；严格边界与预算未知保留。')
    if name=='fit_search': st.caption('TRUE 是零中心误差的充分证明；未找到支持见证不表示通用 fits 不存在。输入无需给出仿射矩阵。')
    if name=='collinear_fits': st.caption('仅共线中心、给定分划与固定横向延拓；全局稳定中心互位后逐点核验严格包含。未找到见证不表示通用 fits 不存在。')
    if name=='four_center_fits': st.caption('四个一般仿射位置 source 中心；全部排列全局互位、共同逐点小位移稳定与完整块严格包含的充分证明。')
    if name=='compatibility': st.caption('只输入完整环境与标签互斥群；自动寻找分划及变换。精确粒度和完整群结论分别核验，未知不表示一般不兼容。')
    if name=='circuit_fits': st.caption('完整4–7有理点、给定分划与秩2中心；所有对应的四点关系下界及所选全局稳定最优变换构成充分证书，仍不判定一般 fits。')
    if name=='translation': st.caption('固定有理正交朝向、显式 source→target 对应，求 Q(source)+平移 的最优最大误差。默认三点非零例；不优化转角或对应。')
    if name=='candidate_acceptance': st.caption('默认 Hall 反例：所有点都有邻居，普通接受为 True，窄接受为 False。只核验给定位置与半径平方。')
    if name=='placed_bottleneck': st.caption('默认三点给定位置例最小瓶颈距离平方为64；保持位置，只自动搜索对应与阈值。')
    if name=='environment_code': st.caption('默认环境两三角形11/22/33与33/44/55重叠于33；子图标签11/22/33/44保持全局编号。代码可在平面定理域外构造，不作仿射等价结论。')
    if name=='ellipse_code2019': st.caption('所有三元组必须非共线；默认非平行四边形有4个精确椭圆、12项带索引比例。平面定理域另行显示，不作仿射等价结论。')
    if name=='proportionality': st.caption('默认重叠部分反例：原文单向比例定义为True，但退化保持与全部面积比例为False。三项分开展示；给定对应失败不表示不存在其他仿射对应。')
    if name=='ellipse_equivalence2019': st.caption('完整两侧各3–8点，所有三元组非共线；自动搜索同一编号双射。代码等价、作者平面定理域与同一映射仿射见证分开；先构造，再显式独立核验下载证书。')
    text = st.text_area('完整精确输入 JSON', value=example,height=350,key='exact_'+name+'_input')
    work = st.number_input('总工作预算', min_value=0, max_value=262144 if name=='futlyar' else 2000000,
                           value=262144 if name=='futlyar' else 2000000,step=1,key=name+'_work')
    seconds = st.number_input('时间预算（秒）', min_value=0,max_value=30,value=10,key=name+'_seconds') if name!='futlyar' else None
    signature = hashlib.sha256(json.dumps([mode, text, work, seconds], ensure_ascii=False).encode()).hexdigest()
    if st.button('搜索并构造椭圆等价证书' if name=='ellipse_equivalence2019' else '计算并独立核验证书', key='exact_compute'):
        try:
            result = compute(text,max_work=int(work),**(dict(seconds=int(seconds)) if seconds is not None else {}))
            st.session_state['exact_result'] = {'signature': signature, 'result': result}
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            st.session_state.pop('exact_result', None)
            st.error('输入或证据错误：' + str(exc))
    saved = st.session_state.get('exact_result')
    if not saved: return
    if saved['signature'] != signature:
        st.warning('输入或预算已改变：旧结果失效，请显式重新计算。')
        return
    result = saved['result']
    if name=='ellipse_equivalence2019':
        if st.button('独立核验下载证书',key='exact_ellipse_equivalence2019_verify',disabled=not result['has_certificate']):
            try:
                checked=verify_ellipse_equivalence2019_payload(result['payload'],max_work=int(work),seconds=int(seconds))
                result['independent_check']=checked['status']
                result['verification_reason']=checked.get('reason','')
            except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
                result['independent_check']='INVALID'
                result['verification_reason']=str(exc)
        st.caption('独立核验按钮只回读下载证书，不重新搜索。无证书时禁用；下载和算法状态不代替独立核验。')
    st.write('数学状态：' + result['status'])
    st.write('序列化后独立核验：' + str(result['independent_check']))
    if family:
        st.write(f"覆盖记录 {result['family_count']}；UNKNOWN {result['unknown_covers']}；完整精确完成 {result['complete']}")
        st.caption('独立核验 True 表示部分或完整结果证据有效；UNKNOWN 原样保留，不等同于全部构造完成。')
    else:
        if result['reason']: st.write('原因：' + result['reason'])
        if result.get('check_note'): st.info(result['check_note'])
        if name=='center':
            st.write('精确域：'+result['profile'])
            st.json(encode_exact(dict(center=result['center'],major_axis_length=result['major_axis_length'])))
            st.caption('点/线给出规范点或线段中心；不声称非退化最小椭圆。UNKNOWN 不表示不存在椭圆。')
        if name=='environment':
            st.write(f"非空部分 {result['parts_count']}；标签互斥组 {result['group_count']}；完整枚举 {result['complete']}")
            st.write('部分独立核验：'+str(result['parts_check'])+'；S* 基数：'+str(result['star_cardinality']))
        if name=='fit_search':
            st.write(f"中心秩 {result['center_rank']}；尝试候选 {result['candidates_tried']}")
            st.caption('通用 fits 仍未判定；仅返回可独立核验的有限充分见证。')
        if name in ('collinear_fits','four_center_fits','circuit_fits'):
            if result['error_squared'] is not None: st.write('精确最大中心误差平方：'+str(result['error_squared']))
            st.caption('非零中心误差仍可得到充分证据；块索引与外部点标签分别保留，通用 fits 仍未判定。')
        if name=='compatibility':
            st.write(f"兼容粒度下界 {result['lower_bound']}；基数上界 {result['upper_bound']}；精确粒度 {result['granularity']}")
            st.write(f"完整群状态：{result['full_group_status']}；独立核验：{result['full_group_check']}；访问候选 {result['candidates_tried']}")
        if name in ('translation','placed_bottleneck'):
            if result['distance_squared'] is not None: st.write('精确最大误差平方：'+str(result['distance_squared']))
            if name=='translation' and result['translation'] is not None: st.json(encode_exact(result['translation']))
        if name=='candidate_acceptance':
            st.write('普通接受：'+str(result['acceptable'])+'；窄接受（双射）：'+str(result['narrow_acceptable']))
            if result['obstruction'] is not None: st.json(encode_exact(result['obstruction']))
        if name in ('environment_code','ellipse_code2019'):
            st.write('来源平面定理域：'+str(result['plane_domain'])+'；不判仿射等价')
            st.write('全局标签：'+str(result['global_labels']))
            st.write('带索引比例项数：'+str(result['ratio_count']))
            if name=='environment_code':
                st.write('子图全局标签：'+str(result['subset_labels']))
                st.write('全环境比例项数：'+str(result['environment_ratio_count'])+'；全环境平面域：'+str(result['environment_plane_domain']))
                st.write('子代码定义：'+str(result['code_defined'])+'；三元组数：'+str(result['triangle_count'])+'；零分母数：'+str(result['zero_denominator_count']))
            else: st.write('精确几何椭圆数：'+str(result['ellipse_count']))
        if name=='proportionality':
            st.write('原文给定对应比例性：'+str(result['paper_proportionality_given_mapping']))
            st.write('退化保持：'+str(result['degeneracy_preserved']))
            st.write('全部面积比例：'+str(result['full_area_proportionality']))
            st.write('平面定理适用：'+str(result['theorem_eligible']))
            st.write('全环境标签：'+str(result['global_labels'])+'；source标签：'+str(result['source_labels'])+'；target标签：'+str(result['target_labels']))
            st.write('比例常数：'+str(result['constant'])+'；三元组账本：'+str(result['ledger_count']))
            st.caption('自对比例1可由同一非零面积导出，但不是环境代码索引项；本例自对数：'+str(result['self_pair_count']))
            st.caption('正负只针对给定对应，不证明不存在其他对应，不作全局非仿射结论。')
        if name=='ellipse_equivalence2019':
            st.write('原文椭圆代码等价：'+str(result['code_equivalent']))
            st.write('作者平面定理域：'+str(result['theorem_domain']))
            st.write('定理层结论：'+result['theorem_conclusion'])
            st.write('同一编号映射仿射见证：'+result['affine_mapping_status'])
            st.write(f"完整比例账本 {result['ratio_count']}；已排除对应 {result['rejected_correspondences']}")
            if result.get('verification_reason'):st.write('独立核验说明：'+result['verification_reason'])
            st.caption('平面域外只报告代码谓词；NOT_ESTABLISHED不否定其他仿射对应。仍非整篇论文、照片或CAD验收。')
        st.caption('算法状态与序列化后独立核验分别展示；未知和未支持范围不能视为已完成。')
    st.download_button('下载完整精确结果', result['payload'], file_name='exact-research-result.json', mime='application/json')
    st.download_button('下载证书与见证', result['witness'], file_name='exact-research-witness.json', mime='application/json')


if __name__ == '__main__': render()
