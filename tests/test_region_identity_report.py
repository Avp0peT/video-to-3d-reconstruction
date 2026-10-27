from integrations.video_masks.region_identity_report import group_splits, parse_points


def test_native_row_binding_keeps_point_groups_separate(tmp_path):
    p=tmp_path/'points3D.txt'
    p.write_text('# native\n1 0 0 1 0 0 0 .2 5 10 6 20\n2 1 0 1 0 0 0 .3 7 30\n3 2 0 1 0 0 0 .1 5 2\n')
    frames=[dict(image_id=i,input_index=j,decode_index=j*17,row_by_track={'9':r},points={'9':[j+1.,j+2.]}) for j,(i,r) in enumerate(((5,10),(6,20),(7,30)))]
    g=group_splits(parse_points(p),frames)
    assert len(g)==1 and g[0]['region_id']==9
    assert [x['point3D_id'] for x in g[0]['point_groups']]==[1,2]
    assert [o['input_index'] for o in g[0]['point_groups'][0]['observations']]==[0,1]
    assert g[0]['identity_status']=='NOT_VERIFIED'
    assert g[0]['split_is_automatic_mismatch_verdict'] is False


def test_single_3d_point_is_not_split():
    frames=[dict(image_id=1,input_index=0,decode_index=0,row_by_track={'7':4},points={'7':[1.,2.]})]
    points=[dict(point3D_id=3,xyz=[0,0,1],error=.1,observations=[(1,4)])]
    assert group_splits(points,frames)==[]
