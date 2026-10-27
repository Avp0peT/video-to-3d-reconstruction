import copy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025 import cover_center_family as mod


LINE = {0:(0,0), 1:(1,0), 2:(2,0), 3:(3,0)}
SQUARE = {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)}
CONTACT = {0:(0,0), 1:(3,0), 2:(0,3), 3:(F(7,4),F(7,4))}


def test_line_complete_all_eight_covers():
    r = mod.build_cover_center_family(LINE)
    assert r['complete'] and r['exact_covers'] == 8 and r['unknown_covers'] == 0
    assert r['expected_candidates'] == len(r['decision_ledger']) == 15
    assert mod.verify_cover_center_family(LINE,r)
    assert r['families'][0]['centers'] == [(F(3,2),0)]
    assert r['families'][0]['cover_size']['lengths'][0]['squared'] == 9


def test_contact3_fallback_full_block_not_support_replacement():
    r = mod.build_cover_center_family(CONTACT)
    row = r['families'][0]
    assert row['status'] == 'EXACT' and row['blocks'][0]['kind'] == 'contact3'
    assert row['blocks'][0]['labels'] == [0,1,2,3]
    assert row['blocks'][0]['certificate']['support_labels'] == [0,1,2]
    assert row['centers'] == [(1,1)] and r['complete']
    assert mod.verify_cover_center_family(CONTACT,r)


def test_contact5_fallback_and_independent_verification(monkeypatch):
    # Five strictly positive contacts with hand-derived optimum unit circle.
    points = {0:(1,0), 1:(F(-3,5),F(4,5)), 2:(F(-3,5),F(-4,5)), 3:(0,1), 4:(0,-1)}
    r = mod.build_cover_center_family(points)
    block = r['families'][0]['blocks'][0]
    assert block['kind'] == 'contact5' and block['labels'] == list(points)
    assert block['center'] == (0,0) and block['certificate']['shape_matrix'] == ((1,0),(0,1))
    def forbidden(*a, **kw): raise AssertionError('constructor used by verification')
    for name in ('complete_input_contact3_ellipse', 'complete_input_contact5_ellipse',
                 'given_cover_supported_block_ostov', 'verify_independent_cover_2d',
                 'enumerate_independent_covers_2d'):
        monkeypatch.setattr(mod, name, forbidden)
    assert mod.verify_cover_center_family(points, r)
    bad = copy.deepcopy(r)
    bad['families'][0]['blocks'][0]['certificate']['contact_weights'] = (F(1,5),)*5
    assert not mod.verify_cover_center_family(points, bad)


def test_contact5_receives_remaining_budget_and_unknown_retained(monkeypatch):
    points = {0:(1,0), 1:(F(-3,5),F(4,5)), 2:(F(-3,5),F(-4,5)), 3:(0,1), 4:(0,-1)}
    received=[]
    def exhausted(*args, **kwargs):
        received.append(kwargs)
        return {'status':'PARTIAL_UNKNOWN', 'work_used':0}
    monkeypatch.setattr(mod, 'complete_input_contact5_ellipse', exhausted)
    r = mod.build_cover_center_family(points, total_max_work=100000, seconds=10)
    assert received and all(0 < row['max_work'] < 100000 and 0 < row['seconds'] < 10 for row in received)
    assert not r['complete'] and r['families'][0]['status'] == 'UNKNOWN'
    assert r['families'][0]['blocks'][0]['reason'] == 'PARTIAL_UNKNOWN'
    assert mod.verify_cover_center_family(points, r)


def test_square_contact4_now_exact_complete():
    r = mod.build_cover_center_family(SQUARE)
    assert r['schema'] == 'cover-center-family-v2'
    assert r['enumeration_complete'] and r['decision_ledger_complete'] and r['complete']
    assert len(r['families']) == 14 and r['exact_covers'] == 14 and r['unknown_covers'] == 0
    assert r['families'][0]['status'] == 'EXACT'
    assert r['families'][0]['blocks'][0]['kind'] == 'contact4'
    assert mod.verify_cover_center_family(SQUARE,r)


@pytest.mark.parametrize('mutation',['omit_row','omit_everywhere','label','certificate','center','size','digest','ledger','complete','float','bool_label'])
def test_distinguishable_tampering(mutation):
    r = mod.build_cover_center_family(LINE)
    bad = copy.deepcopy(r)
    row = bad['families'][0]
    if mutation == 'omit_row': bad['families'].pop(0)
    elif mutation == 'omit_everywhere':
        bad['families'].pop(0);bad['enumeration']['independent_covers'].pop(0)
        bad['enumeration']['rejected_candidates'] += 1;bad['exact_covers'] -= 1
    elif mutation == 'label': row['blocks'][0]['labels'].pop()
    elif mutation == 'certificate': row['blocks'][0]['certificate']['pieces'][0]['size']['squared'] += 1
    elif mutation == 'center': row['blocks'][0]['center'] = (0,0)
    elif mutation == 'size': row['cover_size']['lengths'] = []
    elif mutation == 'digest': bad['source_coordinate_sha256'] = 'fake'
    elif mutation == 'ledger': bad['decision_ledger'].pop()
    elif mutation == 'complete': bad['complete'] = False
    elif mutation == 'float': row['blocks'][0]['certificate']['pieces'][0]['center'] = (1.5,0.)
    elif mutation == 'bool_label': row['blocks'][0]['labels'][0] = False
    assert not mod.verify_cover_center_family(LINE,bad)


def test_unknown_cannot_be_promoted_or_dropped(monkeypatch):
    monkeypatch.setattr(mod, 'complete_input_contact4_ellipse',
        lambda *a, **k: {'status': 'PARTIAL_UNKNOWN', 'work_used': 0})
    r = mod.build_cover_center_family(SQUARE)
    bad = copy.deepcopy(r);bad['complete']=True;bad['status']='COMPLETE'
    assert not mod.verify_cover_center_family(SQUARE,bad)
    bad = copy.deepcopy(r);bad['families'][0]['status']='EXACT'
    assert not mod.verify_cover_center_family(SQUARE,bad)
    assert not mod.verify_cover_center_family({**SQUARE,4:(3,3)},r)


def test_algebraic_root_context_and_certificate_tampering(monkeypatch):
    points = {0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2))}
    r = mod.build_cover_center_family(points)
    assert r['complete'] and r['families'][0]['blocks'][0]['kind'] == 'contact4'
    center = r['families'][0]['centers'][0]
    assert center['expression'] == 'algebraic_vector' and len(center['root']['poly']) == 4
    def forbidden(*a, **k): raise AssertionError('constructor used in independent check')
    monkeypatch.setattr(mod, 'complete_input_contact4_ellipse', forbidden)
    assert mod.verify_cover_center_family(points, r)
    bad = copy.deepcopy(r)
    bad['families'][0]['blocks'][0]['certificate']['contact_weights'][0]['numerator'][0][0] += 1
    assert not mod.verify_cover_center_family(points, bad)
    bad = copy.deepcopy(r)
    bad['schema'] = 'cover-center-family-v1'
    assert not mod.verify_cover_center_family(points, bad)


def test_contact4_receives_remaining_shared_budget(monkeypatch):
    received = []
    def unresolved(*a, **kw):
        received.append(kw)
        return {'status':'PARTIAL_UNKNOWN', 'work_used':17}
    monkeypatch.setattr(mod, 'complete_input_contact4_ellipse', unresolved)
    r = mod.build_cover_center_family(SQUARE, total_max_work=100000, seconds=10)
    assert received and 0 < received[0]['max_work'] < 100000 and 0 < received[0]['seconds'] < 10
    assert r['unknown_covers'] == 1 and not r['complete'] and mod.verify_cover_center_family(SQUARE, r)


def test_verifier_calls_no_solver_hull_or_constructor(monkeypatch):
    r = mod.build_cover_center_family(CONTACT)
    def forbidden(*a,**kw): raise AssertionError('solver called by verifier')
    for name in ('enumerate_independent_covers_2d','verify_independent_cover_2d',
                 'given_cover_supported_block_ostov','complete_input_contact3_ellipse'):
        monkeypatch.setattr(mod,name,forbidden)
    import kozlov_lab.kozlov2025.independent_cover as independent
    monkeypatch.setattr(independent,'hull',forbidden)
    assert mod.verify_cover_center_family(CONTACT,r)


@pytest.mark.parametrize('kw',[{'seconds':0},{'total_max_work':0},{'total_max_work':5},{'max_candidates':1}])
def test_budget_partial_no_completion(kw):
    r = mod.build_cover_center_family(SQUARE,**kw)
    assert not r['complete'] and r['status']=='PARTIAL_UNKNOWN'
    if 'enumeration' in r and 'exact_covers' in r:
        assert len(r['families']) == len(r['enumeration']['independent_covers'])


def test_budget_is_shared_not_per_block(monkeypatch):
    received=[]
    original=mod.given_cover_supported_block_ostov
    def recorded(*a,**kw):
        received.append((kw['max_work'],kw['seconds']))
        return original(*a,**kw)
    monkeypatch.setattr(mod,'given_cover_supported_block_ostov',recorded)
    r=mod.build_cover_center_family(LINE,total_max_work=10000,seconds=10)
    assert r['complete'] and len(received)>1
    assert all(a[0]>b[0] and a[1]>b[1] for a,b in zip(received,received[1:]))
    assert r['work_used'] <= 10000


@pytest.mark.parametrize('points',[{}, {False:(0,0)}, {0:(0,0),1:(0,0)}, {i:(i,0) for i in range(8)}, {0:(float('nan'),0)}])
def test_invalid_complete_input(points):
    assert mod.build_cover_center_family(points)['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize('kw',[{'seconds':True},{'total_max_work':True},{'max_candidates':True},{'seconds':31}])
def test_invalid_budget(kw):
    assert mod.build_cover_center_family(LINE,**kw)['status']=='UNSUPPORTED_INPUT'
