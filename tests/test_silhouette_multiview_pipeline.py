"""Adapter contract tests with synthetic bound native inputs, no real run."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from integrations.mvs_surface import silhouette_multiview_pipeline as p


def fixture(tmp_path,monkeypatch,empty):
    source = tmp_path/'source';source.mkdir()
    input_dir = tmp_path/'inputs';input_dir.mkdir()
    frames,views,maps,bindings = [],[],{},[]
    shape = (12,12)
    for i in range(7):
        image = input_dir/('image'+str(i)+'.png')
        mask = input_dir/('mask'+str(i)+'.png')
        valid = input_dir/('valid'+str(i)+'.png')
        Image.fromarray(np.full((*shape,3),100,np.uint8)).save(image)
        Image.fromarray(np.full(shape,255,np.uint8)).save(mask)
        Image.fromarray(np.full(shape,255,np.uint8)).save(valid)
        dmap = input_dir/('depth'+str(i)+'.dmap');dmap.write_bytes(b'SYNTHETIC'+bytes([i]))
        if i != 4: bindings.append(p.binding(dmap))
        maps[str(dmap)] = SimpleNamespace(view_ids=[0,3,1,2,6] if i == 0 else [i],K=np.array([[20.,0,6],[0,20.,6],[0,0,1]]),R=np.eye(3),C=np.array([.01*i,0,0]),depth=np.full(shape,2.,np.float32),d_min=1.,d_max=4.)
        frames.append(dict(image=dict(global_id=i,width=shape[1],height=shape[0],K=maps[str(dmap)].K.tolist(),R=maps[str(dmap)].R.tolist(),C=maps[str(dmap)].C.tolist()),image_file=p.binding(image),mask=p.binding(mask),validity=p.binding(valid)))
        views.append(dict(global_id=i,decode_index=i*10))
    plan = input_dir/'plan.json';p.save(plan,dict(frames=frames))
    p.save(source/'PREPARED.json',dict(source_plan=str(plan),refs=[p.binding(plan)],views=views))
    p.save(source/'RESULT.json',dict(status='REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED',arms={'WHOLE':dict(execution_ok=True,depth_bindings=bindings)}))
    whole = source/'WHOLE/work';whole.mkdir(parents=True)
    p._write_cloud(whole/'__mvs_dense.ply',np.array([[0.,0.,2.]]),np.array([[40,80,120]],np.uint8))
    spec = input_dir/'roi.json';p.save(spec,dict(decode_index=0,cells=[dict(bounds=[3,3,9,9])]))
    monkeypatch.setattr(p,'read_dmap',lambda path:maps[str(path)])
    monkeypatch.setattr(p,'authenticate_depth',lambda d,image:None)
    def estimate(ref,sources,roi,bounds,check):
        check()
        assert [s['view_id'] for s in sources] == [3,1,2,4,5,6]
        assert not sources[3]['depth_available'] and not sources[3]['depth'].any()
        assert sources[-1]['view_id'] == 6
        assert all('validity' in s for s in sources)
        bounded = np.zeros(shape,bool)
        if not empty: bounded[6,6] = True
        front = bounded.astype(float)*1.8;back=bounded.astype(float)*2.2
        reason = np.zeros(shape,np.uint8);reason[roi]=1;reason[bounded]=5
        return dict(front_depth=front,back_depth=back,interval_width=back-front,bounded_mask=bounded,sampled_mask=roi.copy(),reason=reason,baseline_reason=reason.copy(),stage_counts=[dict(admission_source_ids=[3,1,2],bounded_pixels=int(bounded.sum()))],front_transition_low=front-.01*bounded,front_transition_high=front+.01*bounded,back_transition_low=back-.01*bounded,back_transition_high=back+.01*bounded,counts=dict(roi_pixels=int(roi.sum()),bounded_pixels=int(bounded.sum())),withheld_endpoint_counts={'front':dict(evaluated=int(bounded.sum()))},admission_source_ids=[3,1,2,4,5],withheld_source_id=6)
    monkeypatch.setattr(p,'tighten_silhouette_intervals',estimate)
    def viewer(spec,out,profile):
        out.write_text('<html>synthetic viewer</html>')
        return dict(path=str(out))
    monkeypatch.setattr(p,'export_html',viewer)
    protected = {x:x.read_bytes() for x in [*source.rglob('*'),*input_dir.rglob('*')] if x.is_file()}
    return source,spec,tmp_path/'output',protected


@pytest.mark.parametrize('empty',[False,True])
def test_adapter_preserves_inputs_and_all_arrays(tmp_path,monkeypatch,empty):
    source,spec,out,protected = fixture(tmp_path,monkeypatch,empty)
    result = p.run(source,0,spec,out)
    assert result['geometry_accepted'] is False and result['complete'] is False
    assert result['existing_whole_unchanged'] is True
    assert result['exact_array_readback'] is True
    for file,content in protected.items(): assert file.read_bytes() == content
    assert (out/'whole_cloud.ply').read_bytes() == (source/'WHOLE/work/__mvs_dense.ply').read_bytes()
    scenes = p.load(out/'VIEWER_SPEC.json')['scenes']
    assert len(scenes) == (1 if empty else 3)
    assert (out/'front_midpoints.ply').exists() == (not empty)
    assert (out/'back_midpoints.ply').exists() == (not empty)
    for iid in (3,1,2,4,5,6):
        assert (out/('projection_'+str(iid)+'.png')).exists() == (not empty)
        if not empty:
            counts = result['midpoint_projection_counts'][str(iid)]
            assert counts['front']['evaluated'] == counts['back']['evaluated'] == 1
            assert counts['front']['foreground'] == counts['back']['foreground'] == 1
    if empty: assert result['midpoint_projection_counts'] == {}
    assert (out/'reference.png').exists() and (out/'source_4.png').exists()
    with np.load(out/'SILHOUETTE_INTERVALS.npz',allow_pickle=False) as arrays:
        assert 'front_transition_low' in arrays and 'back_transition_high' in arrays
        assert arrays['bounded_mask'].sum() == (0 if empty else 1)
    if not empty:
        front = p.read_ply(out/'front_midpoints.ply',max_bytes=1<<20,max_vertices=10,max_faces=1)
        np.testing.assert_allclose(np.asarray(front['vertices']).reshape(-1,3),[[0,0,1.8]])
        np.testing.assert_array_equal(np.asarray(front['colors']).reshape(-1,3),[[255,150,30]])
    assert '交互对照' in (out/'index.html').read_text(encoding='utf-8')
    assert p.load(out/'PLAN.json')['withheld_source_id'] == 6


def test_adapter_records_kernel_failure(tmp_path,monkeypatch):
    source,spec,out,protected = fixture(tmp_path,monkeypatch,False)
    def fail(*args,**kwargs): raise RuntimeError('synthetic failure')
    monkeypatch.setattr(p,'tighten_silhouette_intervals',fail)
    with pytest.raises(RuntimeError,match='synthetic failure'): p.run(source,0,spec,out)
    assert p.load(out/'RESULT.json')['status'] == 'FAILED_NOT_ADOPTED'
    for file,content in protected.items(): assert file.read_bytes() == content


def test_protected_output_and_modified_historical_plan_rejected(tmp_path,monkeypatch):
    source,spec,out,_ = fixture(tmp_path,monkeypatch,True)
    with pytest.raises(ValueError,match='fresh disjoint'): p.run(source,0,spec,source/'nested')
    plan = spec.parent/'plan.json';plan.write_text('{}')
    with pytest.raises(ValueError,match='historical source plan changed'): p.run(source,0,spec,out)


def test_actual_kernel_unbounded_never_adds_and_baseline_matches(tmp_path,monkeypatch):
    from integrations.mvs_surface.silhouette_multiview import tighten_silhouette_intervals
    source,spec,out,_ = fixture(tmp_path,monkeypatch,True)
    monkeypatch.setattr(p,'tighten_silhouette_intervals',tighten_silhouette_intervals)
    result = p.run(source,0,spec,out)
    assert result['metrics']['bounded_pixels'] == 0
    assert result['kernel_metadata']['baseline_summary']['bounded'] == 0
    assert not result['clouds']
    assert result['source_depth_available']['4'] is False
    assert len(result['kernel_metadata']['whole_depth_source_classifications']) == 6
    assert len(result['kernel_metadata']['stage_counts']) >= 1


def test_missing_depth_camera_validation_and_mask_binding(tmp_path,monkeypatch):
    source,spec,out,_ = fixture(tmp_path,monkeypatch,True)
    mask = spec.parent/'mask4.png'
    mask.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mask binding changed'):
        p.run(source,0,spec,out)


def test_missing_depth_uses_authenticated_plan_camera(tmp_path,monkeypatch):
    source,spec,out,_ = fixture(tmp_path,monkeypatch,True)
    plan_path = spec.parent/'plan.json'
    plan = p.load(plan_path)
    plan['frames'][4]['image']['K'][0][0] = -1.
    p.save(plan_path,plan)
    prepared = p.load(source/'PREPARED.json')
    prepared['refs'] = [p.binding(plan_path)]
    p.save(source/'PREPARED.json',prepared)
    with pytest.raises(ValueError,match='invalid calibration'):
        p.run(source,0,spec,out)
