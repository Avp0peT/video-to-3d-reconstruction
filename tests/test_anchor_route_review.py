import json
from pathlib import Path
import pytest
from integrations.demo_pipeline.anchor_route_review import review_anchor_routes, _clouds
from integrations.demo_pipeline.temporal_review import fingerprint


def test_empty_measured_chain_produces_honest_automatic_report(tmp_path):
    plan=dict(routes=[],decode_indices=[],omitted_routes=[])
    result=review_anchor_routes(plan,dict(records=[]),[],[],{},tmp_path/'report',[],None,{},
                               dict(joins=[]))
    assert result['status']=='ANCHOR_ROUTE_REVIEW_COMPLETE'
    assert result['numeric_groups']==0 and result['exported_conditional_points']==0
    assert not result['geometry_accepted'] and result['flow_calls']==0
    assert result['comparison']['newly_linked_group_ids']==[]
    assert all(fingerprint(r['path'])==r for r in result['source_bindings'])
    assert not (tmp_path/'report/viewer.html').exists()
    assert (tmp_path/'report/projection/index.html').is_file()


def test_only_classified_conditional_points_exported_by_model_and_tier(tmp_path):
    def row(model,tid,tier):
        return dict(model_key=model,group_id=f'{model}:{tid}:0',track_id=tid,identity_segment=0,
            tier=tier,xyz=[1.,2.,3.],sources=[],geometry_accepted=False,native_injected=False)
    rows=[row('a',1,'STRICT_CONDITIONAL'),row('b',1,'RELAXED_CONDITIONAL'),row('a',2,'REJECTED')]
    (tmp_path/'GEOMETRY.json').write_text(json.dumps(dict(classification=dict(rows=rows))))
    artifacts,clouds=_clouds(tmp_path,dict(rows=rows),lambda:None)
    assert len(clouds)==2 and sum(r['vertices'] for r in clouds)==2
    assert all(r['readback'] and r['faces']==0 and not r['geometry_accepted'] for r in clouds)
    assert {r['model_key'] for r in clouds}=={'a','b'}
    assert (tmp_path/'viewer.html') in artifacts
    assert all(fingerprint(r['binding']['path'])==r['binding'] for r in clouds)


def test_failed_stage_not_published_as_complete(tmp_path):
    def budget():raise TimeoutError('shared deadline')
    with pytest.raises(TimeoutError,match='shared deadline'):
        review_anchor_routes(dict(routes=[]),dict(records=[]),[],[],{},tmp_path/'report',[],None,{},dict(joins=[]),budget=budget)
    failure=json.loads((tmp_path/'report/FAILURE.json').read_text())
    assert failure['status']=='RESOURCE_LIMIT'
    assert not (tmp_path/'report/RESULT.json').exists()


def test_actual_route_wiring_to_geometry_and_images(tmp_path):
    from test_anchor_route_tracking import fixture, CV
    from integrations.demo_pipeline.anchor_route_tracking import plan_anchor_routes
    from PIL import Image
    pop,frames,cameras,ledger,decoded,masks=fixture(3)
    for record in ledger['records']:
        record.update(edge_ordinal=0,verified_legs=2,observations=[dict(decode_index=i,
            raw_xy=[8.5+i,10.5],width=32,height=32,
            original_mask_status='OK' if i%2==0 else 'UNKNOWN_NOT_SEGMENTED') for i in (2,1,0)])
    for index,frame in decoded.items():
        path=tmp_path/f'frame{index}.png'
        Image.fromarray(frame['gray']).convert('RGB').save(path)
        frame['path']=str(path)
    # The test camera/flow combination is allowed to be geometrically rejected.
    # Every measured route must nevertheless reach the real numerical stage.
    result=review_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,
        tmp_path/'report',[],CV(),cameras,dict(joins=[]))
    assert result['measurement_counts']=={'MEASURED_THREE_VIEW_ANCHOR':3}
    assert result['numeric_groups']==3
    assert (tmp_path/'report/anchors_000.png').is_file()
    assert all(fingerprint(r['path'])==r for r in result['source_bindings'])
