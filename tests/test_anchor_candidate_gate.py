"""Numerical candidate contracts with exact geometry; no native or model work."""
from copy import deepcopy
import json
from collections import Counter,defaultdict,deque
import numpy as np
import pytest
from integrations.demo_pipeline.original_anchor_review import PROFILE as SOURCE_PROFILE
from integrations.demo_pipeline.anchor_candidate_gate import classify_original_candidates, _tier


def seal_batches(source):
    """Seal a mutated synthetic diagnostic with the production partition contract."""
    byid={g['group_id']:g for g in source['groups']};queue=[]
    for multiview in (True,False):
        buckets=defaultdict(deque)
        for g in sorted(source['groups'],key=lambda g:(g['model_key'],g['track_id'],g['identity_segment'])):
            n=len(g['admitted_observations'])
            if 2<=n<=12 and (n>=3)==multiview:buckets[g['model_key']].append(g['group_id'])
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:queue.append(buckets[key].popleft())
    selected=queue[:96]
    for ident in queue[96:]:byid[ident]['status']='NOT_EVALUATED_GROUP_BUDGET'
    source['selected_groups']=selected
    source['omitted_groups']=[dict(group_id=g['group_id'],status=g['status']) for g in source['groups'] if g['group_id'] not in selected]
    source['selection']=dict(policy=SOURCE_PROFILE['selection'],eligible_groups=len(queue),batch_size=32,max_batches=3,max_groups=96)
    source['batches']=[]
    for start in range(0,len(selected),32):
        ids=selected[start:start+32];groups=[byid[i] for i in ids]
        source['batches'].append(dict(batch_id=start//32,group_ids=ids,status='COMPLETE',evaluated_groups=len(ids),
            group_statuses=dict(Counter(g['status'] for g in groups)),loo_checks=sum(len(g['leave_one_view_out']) for g in groups)))
    source['counts'].update(groups=len(byid),selected_groups=len(selected),omitted_groups=len(byid)-len(selected),
        statuses=dict(Counter(g['status'] for g in byid.values())),batches_completed=len(source['batches']))
    return source


def diagnostics(n=4,k=0.):
    xyz=np.array([.1,.05,5.]);observations=[];sources=[]
    for i in range(n):
        index=10+i*9;t=np.array([-.6+i*.4,0.,0.]);local=xyz+t;uv=local[:2]/local[2]
        raw=(100*uv*(1+k*np.dot(uv,uv))+[50,50]).tolist()
        observations.append(dict(input_index=index,frame_id=f'f{index}',image_id=i+1,raw_pixels=raw,
            camera_model='SIMPLE_RADIAL',params=[100.,50.,50.,k],R=np.eye(3).tolist(),t=t.tolist(),width=100,height=100))
        sources.append(dict(input_index=index,frame_id=f'f{index}',track_id=7,identity_segment=0,
            source_xy=raw.copy(),status='ADMITTED_ORIGINAL_ANCHOR',reasons=[]))
    def fitted(rows):return dict(status='FITTED',converged=True,xyz=xyz.tolist(),train_image_ids=[o['image_id'] for o in rows])
    folds=[dict(heldout_input_index=o['input_index'],heldout_image_id=o['image_id'],
        training_input_indices=[r['input_index'] for r in observations if r is not o],fit=fitted([r for r in observations if r is not o]),geometry_acceptance=False,native_injected=False,identity_verified=False) for o in observations] if n>=3 else []
    group=dict(group_id='0:7:0',model_key='0',track_id=7,identity_segment=0,status='CONDITIONAL_DIAGNOSTIC_EVALUATED',
        observations=sources,admitted_observations=observations,fit=fitted(observations),leave_one_view_out=folds,
        strict_radial_domain_check=[],geometry_acceptance=False,native_injected=False,identity_verified=False)
    return seal_batches(dict(status='ORIGINAL_ANCHOR_DIAGNOSTICS_COMPLETE',profile=deepcopy(SOURCE_PROFILE),groups=[group],selected_groups=['0:7:0'],omitted_groups=[],
        counts=dict(groups=1,selected_groups=1,omitted_groups=0,statuses={group['status']:1}),geometry_acceptance=False,native_injected=False,identity_verified=False))


@pytest.mark.parametrize('n,k',[(2,0.),(3,0.),(4,.2),(4,-.2)])
def test_exact_original_radial_candidates_and_input_unchanged(n,k):
    source=diagnostics(n,k);frozen=deepcopy(source)
    result=classify_original_candidates(source);row=result['rows'][0]
    assert row['tier']==('TWO_VIEW_PROVISIONAL' if n==2 else 'STRICT_CONDITIONAL')
    assert len(row['metrics']['loo'])==(n if n>=3 else 0)
    assert row['metrics']['max_raw_error_px']<1e-8
    assert all(row[key] is False and result[key] is False for key in ('geometry_accepted','native_injected','identity_verified'))
    assert source==frozen
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('n,error,loo,angle,tier',[
    (3,2.,3.,2.,'STRICT_CONDITIONAL'),(3,2.+1e-9,3.,2.,'RELAXED_CONDITIONAL'),
    (3,2.,3.+1e-9,2.,'RELAXED_CONDITIONAL'),(3,2.,3.,2.-1e-9,'RELAXED_CONDITIONAL'),
    (3,4.,6.,1.,'RELAXED_CONDITIONAL'),(3,4.+1e-9,6.,1.,'REJECTED'),
    (3,4.,6.+1e-9,1.,'REJECTED'),(3,4.,6.,1.-1e-9,'REJECTED'),
    (2,4.,None,1.,'TWO_VIEW_PROVISIONAL'),(2,4.+1e-9,None,1.,'REJECTED'),
])
def test_threshold_boundaries(n,error,loo,angle,tier):assert _tier(n,error,loo,angle)==tier


@pytest.mark.parametrize('kind',['full_unconverged','loo_unconverged','fold_duplicate','fold_missing','train_changed',
    'full_train_changed','bad_camera','xy_changed','loo_bad_depth','source_rights','summary_forgery'])
def test_reject_counterexamples_without_trusting_saved_summary(kind):
    source=diagnostics();g=source['groups'][0]
    if kind=='full_unconverged':g['fit']['converged']=False
    elif kind=='loo_unconverged':g['leave_one_view_out'][0]['fit']['converged']=False
    elif kind=='fold_duplicate':g['leave_one_view_out'][1]=deepcopy(g['leave_one_view_out'][0])
    elif kind=='fold_missing':g['leave_one_view_out'].pop()
    elif kind=='train_changed':g['leave_one_view_out'][0]['training_input_indices'].append(10)
    elif kind=='full_train_changed':g['fit']['train_image_ids'].reverse()
    elif kind=='bad_camera':g['admitted_observations'][0]['R'][0][0]=2.
    elif kind=='xy_changed':g['observations'][0]['source_xy'][0]+=1.
    elif kind=='loo_bad_depth':g['leave_one_view_out'][0]['fit']['xyz']=[0.,0.,-5.]
    elif kind=='source_rights':g['geometry_acceptance']=True
    else:
        g['admitted_observations'][0]['raw_pixels'][0]+=8.
        g['observations'][0]['source_xy'][0]+=8.
        g['evaluation']=dict(status='EVALUATED',max=0.,rms=0.)
        g['observability']=dict(status='DIAGNOSTIC_EVALUATED',rank=3)
        g['radial_domains']=[dict(all_observations_invertible=True,max_ideal_radius=999)]
    # Keep batch accounting consistent so this counterexample reaches the
    # per-point missing-fold check; batch-count tampering has separate tests.
    if kind=='fold_missing':seal_batches(source)
    result=classify_original_candidates(source)
    assert result['rows'][0]['tier']=='REJECTED' and result['rows'][0]['xyz'] is None


def test_partition_and_count_tamper_rejected_globally():
    source=diagnostics();source['counts']['groups']=2
    with pytest.raises(ValueError):classify_original_candidates(source)
    source=diagnostics();source['omitted_groups']=[dict(group_id='0:7:0',status='omitted')]
    with pytest.raises(ValueError):classify_original_candidates(source)


def test_budget_timeout_propagates():
    def budget():raise TimeoutError('elapsed')
    with pytest.raises(TimeoutError):classify_original_candidates(diagnostics(),budget)


def test_not_selected_retains_budget_without_numeric_work(monkeypatch):
    from integrations.demo_pipeline import anchor_candidate_gate as gate
    source=diagnostics();g=source['groups'][0]
    g['admitted_observations']=g['admitted_observations']*4
    g['status']='NOT_EVALUATED_OBSERVATION_BUDGET'
    source['selected_groups']=[];source['omitted_groups']=[dict(group_id=g['group_id'],status=g['status'])]
    source['counts']=dict(groups=1,selected_groups=0,omitted_groups=1,statuses={g['status']:1})
    seal_batches(source)
    monkeypatch.setattr(gate,'_measure',lambda *a: (_ for _ in ()).throw(AssertionError('no numeric work')))
    result=classify_original_candidates(source)
    assert result['rows'][0]['tier']=='NOT_EVALUATED'
    assert result['rows'][0]['reasons']==['NOT_EVALUATED_OBSERVATION_BUDGET']


def batch_diagnostics(total=100):
    source=diagnostics(2);base=source['groups'][0];source['groups']=[]
    for i in range(total):
        g=deepcopy(base);g.update(model_key=str(i%2),track_id=i//2,group_id=f'{i%2}:{i//2}:0')
        for row in g['observations']:row['track_id']=g['track_id']
        source['groups'].append(g)
    return seal_batches(source)


def test_gate_rebuilds_three_batches_96_cap_without_numeric_work(monkeypatch):
    from integrations.demo_pipeline import anchor_candidate_gate as gate
    source=batch_diagnostics();calls=[]
    def classify(group,budget):
        calls.append(group['group_id'])
        return dict(group_id=group['group_id'],tier='REJECTED',xyz=None)
    monkeypatch.setattr(gate,'_classify',classify)
    got=gate.classify_original_candidates(source)
    assert len(calls)==96 and len(got['omitted_groups'])==4
    assert [len(b['group_ids']) for b in got['batches']]==[32,32,32]
    assert got['counts']['batches_completed']==3
    assert set(calls)==set(source['selected_groups'])


@pytest.mark.parametrize('kind',['missing','reordered','failed','evaluated','status_counts','loo','duplicate_batch',
    'completed_count','selection','bool_status_counter','bool_selection','unfinished_group','over96'])
def test_gate_batch_tamper_refused_before_numeric(kind,monkeypatch):
    from integrations.demo_pipeline import anchor_candidate_gate as gate
    source=batch_diagnostics()
    if kind=='missing':source['batches'].pop()
    elif kind=='reordered':source['batches'][0]['group_ids'].reverse()
    elif kind=='failed':source['batches'][0]['status']='FAILED'
    elif kind=='evaluated':source['batches'][0]['evaluated_groups']=31
    elif kind=='status_counts':source['batches'][0]['group_statuses']={}
    elif kind=='loo':source['batches'][0]['loo_checks']=1
    elif kind=='duplicate_batch':source['batches'][1]['batch_id']=0
    elif kind=='completed_count':source['counts']['batches_completed']=2
    elif kind=='selection':source['selection']['eligible_groups']=99
    elif kind=='bool_status_counter':
        source=diagnostics(2);source['batches'][0]['group_statuses']={'CONDITIONAL_DIAGNOSTIC_EVALUATED':True}
    elif kind=='bool_selection':source=diagnostics(2);source['selection']['eligible_groups']=True
    elif kind=='unfinished_group':
        source['groups'][0]['status']='READY_FOR_NUMERIC_DIAGNOSTIC';seal_batches(source)
    else:
        source['selected_groups'].append(source['omitted_groups'].pop()['group_id'])
        source['counts'].update(selected_groups=97,omitted_groups=3)
    monkeypatch.setattr(gate,'_classify',lambda *args:pytest.fail('partition rejected before numerics'))
    with pytest.raises(ValueError):gate.classify_original_candidates(source)
