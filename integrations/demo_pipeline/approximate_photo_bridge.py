"""Bounded engineering bridge from frozen photo tracks to S-1996 AlgT'.

Existing COLMAP correspondence and calibration are prerequisites. Neither TLS
nor rectification is claimed as Kozlov's original perspective reconstruction.
"""
from fractions import Fraction
import html
import json
from pathlib import Path
import numpy as np

from . import parallel_photo_audit as prior
from kozlov_lab.kozlov2025.exact_center import encode_exact
from kozlov_lab.stereo.approximate_parallel import (
    fit_affine_flat, evaluate_affine_flat, build_exact_surrogate)
from kozlov_lab.stereo.baseline import radial_diagnostics as radial
from kozlov_lab.stereo.baseline import rectified_parallel as rect


def json_value(value):
    """Mixed numeric ledger: exact fractions tagged, floats remain diagnostics."""
    if isinstance(value, Fraction):
        return {'numerator': value.numerator, 'denominator': value.denominator}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write(path, value):
    prior.write_json(path, json_value(value))


def summary_stats(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty scalar diagnostic required')
    return dict(count=len(a), rms=float(np.sqrt(np.mean(a*a))),
                median_abs=float(np.median(np.abs(a))), max_abs=float(np.max(np.abs(a))))


def split_ids(common, six):
    if (len(six) != 6 or len(set(six)) != 6 or sorted(common) != common
            or len(set(common)) != len(common) or not set(six).issubset(common)):
        raise ValueError('complete sorted IDs and frozen six required')
    remainder = [i for i in common if i not in six]
    train, validation = remainder[::2], remainder[1::2]
    if len(train) < 5 or len(validation) < 5:
        raise ValueError('at least five training and five held-out tracks required')
    return train, validation


def read_calibration(camera_file, image_file):
    r = prior.Reader(camera_file)
    count, = r.take('<Q')
    if not 1 <= count <= 256:
        raise ValueError('bounded native camera count required')
    cameras = {}
    for _ in range(count):
        ident, model, width, height = r.take('<iiQQ')
        if ident <= 0 or ident in cameras or model not in (0, 2) or not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ValueError('unique SIMPLE_RADIAL/SIMPLE_PINHOLE cameras of bounded size required')
        native_params = r.take('<ddd' if model == 0 else '<dddd')
        # A pinhole is exactly the zero-distortion algebraic case; preserve
        # its native model/three parameters alongside the internal radial tuple.
        params = native_params + (0.,) if model == 0 else native_params
        k = radial.simple_radial_intrinsics(params)
        corners = np.array(((0,0),(width,0),(0,height),(width,height)), float)
        radius = 2*float(np.max(np.linalg.norm((corners-k[:2,2])/k[0,0], axis=1)))
        cameras[ident] = dict(params=params, native_model_id=model, native_params=native_params, K=k, width=width, height=height, radius=radius)
    r.finish()
    r = prior.Reader(image_file)
    count, = r.take('<Q')
    if not 2 <= count <= 256:
        raise ValueError('bounded registered image count required')
    poses = {}
    for _ in range(count):
        row = r.take('<idddddddi')
        ident, camera_id = row[0], row[-1]
        if ident in poses or camera_id not in cameras:
            raise ValueError('image pose identity/camera mismatch')
        rotation = radial.quaternion_wxyz_to_rotation(row[1:5])
        translation = np.array(row[5:8], dtype=float)
        if not np.isfinite(translation).all():
            raise ValueError('finite translation required')
        name = r.name()
        n, = r.take('<Q')
        if n > 100000:
            raise ValueError('observation count exceeds bound')
        for _ in range(n):
            r.take('<ddq')
        poses[ident] = dict(R=rotation, t=translation, name=name, camera_id=camera_id)
    r.finish()
    return cameras, poses


def verify_binding(ref):
    if prior.binding(ref['path']) != ref:
        raise ValueError('frozen input binding drift: '+ref['path'])


def _exact(first, second, output, stem):
    result, encoded, check = prior._run_exact(first, second)
    write(output / (stem+'_EXACT.json'), encoded)
    if result.get('certificate') is not None and check.get('status') != 'VERIFIED':
        raise ValueError('independent exact certificate verification failed')
    return result, dict(status=result['status'], reason=result['reason'],
                        algorithm_called=result.get('algorithm_called'), independent=check,
                        certificate_file=stem+'_EXACT.json')


def project_certificate(result, first, second):
    """Use AlgT's actual body/projections; do not treat its gauge as world XYZ."""
    proof = result['certificate']['candidate']['certificate']
    body = dict(proof['body'])
    projections = proof['projections']
    projected = []
    for matrix, wanted in zip(projections, (first, second)):
        view = {}
        for label in sorted(wanted):
            homogeneous = body[label] + (Fraction(1),)
            xy = tuple(sum(a*b for a,b in zip(row, homogeneous)) for row in matrix)
            if xy != wanted[label]:
                raise ValueError('AlgT representative does not reproduce surrogate')
            view[label] = xy
        projected.append(view)
    if len(projected) != 2:
        raise ValueError('two certificate projections required')
    return body, projected


def write_ply(path, labels, xyz, *, source='ENGINEERING_CANDIDATE_NOT_ACCEPTED'):
    if np.asarray(xyz).shape != (len(labels), 3) or not np.isfinite(xyz).all():
        raise ValueError('complete finite candidate coordinates required')
    text = ('ply\nformat ascii 1.0\ncomment '+source+'\n'
            'comment six scene tracks; scale unknown; no faces\n'
            f'element vertex {len(labels)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
    text += ''.join(' '.join(format(float(v), '.17g') for v in row)+'\n' for row in xyz)
    with path.open('x', encoding='ascii') as stream:
        stream.write(text)
    loaded = np.array([[float(v) for v in row.split()]
                       for row in path.read_text(encoding='ascii').split('end_header\n',1)[1].splitlines()])
    if not np.array_equal(loaded, xyz):
        raise ValueError('PLY numeric byte serialization readback failed')
    return prior.binding(path)


def read_reference_xyz(path, labels):
    """Read old multi-view XYZ only after building candidate; never a fit input."""
    r = prior.Reader(path)
    count, = r.take('<Q')
    if not 1 <= count <= 1000000:
        raise ValueError('bounded reference point count required')
    found = {}
    for _ in range(count):
        row = r.take('<QdddBBBd')
        n, = r.take('<Q')
        if n > 100000:
            raise ValueError('bounded reference track length required')
        for _ in range(n):
            r.take('<ii')
        if row[0] in labels:
            if row[0] in found:
                raise ValueError('duplicate reference point ID')
            found[row[0]] = row[1:4]
    r.finish()
    if set(found) != set(labels):
        raise ValueError('missing reference XYZ')
    a = np.array([found[i] for i in labels])
    if not np.isfinite(a).all():
        raise ValueError('nonfinite reference XYZ')
    return a


def experiment(prior_input, output):
    prior_input = Path(prior_input).resolve(strict=True)
    frozen = prior.read_json(prior_input)
    if frozen.get('schema') != 'photo-parallel-selected-observations-v1':
        raise ValueError('previous exact photo audit INPUT required')
    refs = frozen['source_bindings']
    for ref in refs:
        verify_binding(ref)
    by_name = {}
    seen_paths = set()
    for ref in refs:
        path = Path(ref['path']).resolve(strict=True)
        if path in seen_paths:
            raise ValueError('duplicate input binding path')
        seen_paths.add(path)
        if path.name in ('run.json','images.bin','points3D.bin','cameras.bin'):
            if path.name in by_name:
                raise ValueError('duplicate native input binding name')
            by_name[path.name] = path
    if len(by_name) != 4:
        raise ValueError('unique native/run input bindings required')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('new output required; no overwrite')
    if output.is_relative_to(by_name['run.json'].parent) or output.is_relative_to(prior_input.parent):
        raise ValueError('output must be separate from immutable prior runs')
    images, points = prior.read_native(by_name['images.bin'], by_name['points3D.bin'])
    pair, common, _ = prior.choose_pair(images, points)
    six = frozen['point_ids']
    if list(pair) != frozen['image_pair'] or six != common[:6] or len(common) != frozen['common_tracks']:
        raise ValueError('previous view/track selection changed')
    # Exact old six observations, not just count or identity.
    if type(frozen.get('observations')) is not list or len(frozen['observations']) != 6:
        raise ValueError('frozen six ledger must contain exactly six observations')
    for item, pid in zip(frozen['observations'], six):
        if item['point3D_id'] != pid or len(item['observations']) != 2:
            raise ValueError('frozen six ledger mismatch')
        for old, image_id in zip(item['observations'], pair):
            index = points[pid]['track'][image_id]
            xy = images[image_id]['observations'][index][:2]
            if (old['image_id'] != image_id or old['point2D_index'] != index
                    or type(old.get('xy')) is not list or len(old['xy']) != 2
                    or any(type(v) not in (int,float) for v in old['xy'])
                    or old['xy'] != list(xy)
                    or old['binary64_hex'] != [v.hex() for v in xy]):
                raise ValueError('original six coordinates changed')
    train, validation = split_ids(common, six)
    cameras, poses = read_calibration(by_name['cameras.bin'], by_name['images.bin'])
    selected = [cameras[poses[i]['camera_id']] for i in pair]
    p = [poses[i] for i in pair]
    scale = float(max(max(c['width'], c['height']) for c in selected))
    raw = np.array([[v for i in pair for v in images[i]['observations'][points[pid]['track'][i]][:2]] for pid in common])
    row_for = {pid:i for i,pid in enumerate(common)}
    positions = lambda ids: [row_for[i] for i in ids]
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='approximate-photo-bridge-plan-v1', prior_input=prior.binding(prior_input),
        source_bindings=refs, image_pair=pair, six_point_ids=six, common_ids=common,
        training_ids=train, validation_ids=validation, split='remove six, sort IDs, alternate; no geometry reselection',
        tls_scale=scale, tls_grid_bits=18, tls_min_support_ratio=1e-6,
        rectified_scale=1.0, rectified_grid_bits=18,
        rectified_coordinates='(rectified_pixels - Krect principal point) / Krect focal',
        epipolar_correction='mean of the two rectified y values; x unchanged before fixed quantization',
        inverse_max_iterations=128, inverse_tolerance_pixels=1e-8,
        calibrated_numeric_roundtrip_tolerance_pixels=1e-7,
        ideal_radius_policy='twice maximum normalized image-corner radius; refuse nonmonotone domain',
        calibration=[dict(image_id=i, pose=poses[i], camera=c) for i,c in zip(pair,selected)],
        quality_acceptance_threshold=None, ground_truth_available=False,
        calibration_fit_used_heldout_tracks=True, real_model_or_gpu_execution=False)
    write(output/'PLAN.json',plan)  # freeze choices before numeric fitting
    try:
        corrected = [radial.undistort_simple_radial(raw[:,j*2:j*2+2], c['params'],
                     max_ideal_radius=c['radius']) for j,c in enumerate(selected)]
        ideal = np.column_stack([c['ideal_pixels'] for c in corrected])
        tls_results = {}
        for domain, data in (('raw',raw),('undistorted',ideal)):
            fit = fit_affine_flat(data[positions(train)], scale=scale)
            entry = dict(fit=fit)
            if fit['status'] == 'ENGINEERING_APPROXIMATION_FITTED':
                entry['heldout'] = evaluate_affine_flat(fit['model'],data[positions(validation)])
                entry['six_heldout'] = evaluate_affine_flat(fit['model'],data[positions(six)])
                surrogate = build_exact_surrogate(fit['model'],data[positions(six)],six)
                write(output/(domain+'_TLS_SURROGATE.json'),surrogate)
                entry['surrogate_status'] = surrogate['status']
                if surrogate['status'] == 'ENGINEERING_APPROXIMATION_SURROGATE':
                    _, entry['exact'] = _exact(surrogate['source_xy'],surrogate['target_xy'],output,domain+'_TLS')
                    entry['six_max_change_pixels'] = surrogate['max_change_pixels']
            tls_results[domain] = entry
        write(output/'TLS_DIAGNOSTICS.json',tls_results)
        f = radial.fundamental_from_poses(selected[0]['K'],p[0]['R'],p[0]['t'],selected[1]['K'],p[1]['R'],p[1]['t'])
        sampson = radial.sampson_distances(ideal[:,:2],ideal[:,2:],f)
        stereo = rect.calibrated_rectification(selected[0]['K'],p[0]['R'],p[0]['t'],selected[1]['K'],p[1]['R'],p[1]['t'])
        left = rect.apply_homography(ideal[:,:2],stereo['homography1'])
        right = rect.apply_homography(ideal[:,2:],stereo['homography2'])
        vertical = left[:,1]-right[:,1]
        write(output/'OBSERVATIONS.json',dict(point_ids=common, raw_pixels=raw, ideal_pixels=ideal,
            rectified_first_pixels=left,rectified_second_pixels=right,
            inverse_forward_residual_pixels=[c['forward_residual_pixels'] for c in corrected],
            sampson_distance_pixels=sampson['distance'],vertical_difference_pixels=vertical,
            calibration_independent_ground_truth=False))
        write(output/'RECTIFICATION.json',stereo)
        surrogate = rect.quantize_rectified_six(left[positions(six)],right[positions(six)],stereo['intrinsics_rectified'],scale=1.0)
        first, second = dict(zip(six,surrogate['first'])),dict(zip(six,surrogate['second']))
        write(output/'RECTIFIED_SURROGATE.json',dict(point_ids=six,**surrogate))
        actual, exact_summary = _exact(first,second,output,'RECTIFIED')
        candidate = dict(status='NOT_CONSTRUCTED', reason=actual['status'],scale_status='unknown')
        if actual['status'] == 'CONSTRUCTED':
            body, projected = project_certificate(actual,first,second)
            pixel_views = [rect.surrogate_to_rectified_pixels([view[i] for i in six],stereo['intrinsics_rectified'],scale=1.0) for view in projected]
            world = rect.rectified_to_world(*pixel_views,stereo)
            xyz = world['world_xyz']
            per_view = []
            for index,(camera,pose) in enumerate(zip(selected,p)):
                camera_xyz = xyz @ pose['R'].T + pose['t']
                if np.any(camera_xyz[:,2] <= 0):
                    raise ValueError('recovered point behind original camera')
                homogeneous = camera_xyz @ camera['K'].T
                ideal_reproject = homogeneous[:,:2]/homogeneous[:,2,None]
                rectified_readback = rect.apply_homography(ideal_reproject,stereo['homography'+str(index+1)])
                readback_error = float(np.max(np.abs(rectified_readback-pixel_views[index])))
                if readback_error > 1e-7:
                    raise ValueError('source-camera calibrated geometric roundtrip failed')
                raw_reproject = radial.distort_simple_radial(ideal_reproject,camera['params'],max_ideal_radius=camera['radius'])
                difference = raw_reproject-raw[positions(six),index*2:index*2+2]
                per_view.append(dict(image_id=pair[index],raw_reprojection_pixels=raw_reproject,
                    residual_vectors_pixels=difference,residual_norms_pixels=np.linalg.norm(difference,axis=1),
                    stats=summary_stats(np.linalg.norm(difference,axis=1)),
                    calibrated_roundtrip_max_abs_pixels=readback_error))
            ply = write_ply(output/'calibrated_six_points.ply',six,xyz)
            reference_xyz = read_reference_xyz(by_name['points3D.bin'],six)
            reference_ply = write_ply(output/'old_colmap_six_points.ply',six,reference_xyz,
                                      source='EXISTING_COLMAP_REFERENCE_NOT_GROUND_TRUTH')
            comparison = dict(point_ids=six,old_colmap_xyz=reference_xyz,new_bridge_xyz=xyz,
                differences=xyz-reference_xyz,distances_unknown_units=np.linalg.norm(xyz-reference_xyz,axis=1),
                stats_unknown_units=summary_stats(np.linalg.norm(xyz-reference_xyz,axis=1)),
                reference_read_after_candidate=True,reference_is_ground_truth=False,
                old_method='existing multi-view COLMAP',new_method='two-view calibrated rectified surrogate + AlgT + inverse disparity',
                method_workloads_equal=False,scale_status='unknown',old_ply=reference_ply,new_ply=ply)
            write(output/'POINT_COMPARISON.json',comparison)
            write(output/'CALIBRATED_POINTS.json',dict(point_ids=six,world_xyz=xyz,
                affine_gauge_body_exact=encode_exact(tuple(body.items())),projection_readback_exact=True,
                rectified_projection_pixels=pixel_views,disparity_pixels=world['disparity_pixels'],
                reprojection=per_view,point_scope='six scene tracks; foreground membership not certified',
                scale_status='unknown',quality_status='NOT_EVALUATED',ply=ply))
            candidate = dict(status='CALIBRATED_SIX_POINT_CANDIDATE',vertex_count=6,face_count=0,
                path=ply['path'],ply_readback=True,projection_readback_exact=True,
                calibrated_geometry_readback=True,old_new_distance_unknown_units=comparison['stats_unknown_units'],
                scale_status='unknown',quality_status='NOT_EVALUATED',reprojection=[v['stats'] for v in per_view])
        for ref in refs+[plan['prior_input']]:
            verify_binding(ref)
        condensed_tls = {}
        for domain, entry in tls_results.items():
            condensed_tls[domain] = dict(status=entry['fit']['status'],
                train_rms_pixels=entry['fit'].get('training_rms_pixels'),
                heldout_rms_pixels=entry.get('heldout',{}).get('rms_pixels'),
                six_heldout_rms_pixels=entry.get('six_heldout',{}).get('rms_pixels'),
                exact_surrogate=entry.get('exact'),quality_threshold=None)
        result = dict(schema='approximate-photo-bridge-result-v1',status='EXPERIMENT_COMPLETE',
            source_cameras={str(cid):dict(native_model_id=c['native_model_id'],native_params=list(c['native_params'])) for cid,c in cameras.items()},
            engineering_extension=True,source_unchanged=True,image_pair=pair,point_ids=six,
            common_count=len(common),train_count=len(train),validation_count=len(validation),
            tls=condensed_tls,sampson=summary_stats(sampson['distance']),
            rectified_vertical=summary_stats(vertical),rectified_exact=exact_summary,candidate=candidate,
            author_algorithm_scope='actual AlgT prime both views on six modified exact correspondences',
            uses_existing_colmap_calibration_and_correspondence=True,independent_new_reconstruction=False,
            physical_accuracy=None,model_gain=None,mesh_created=False,cad_created=False,
            limits=['Original raw observations have not become exactly parallel.',
                'TLS holdout is for the new fit only; existing calibration used these tracks.',
                'Rectification/averaging/quantization/inverse disparity are project engineering additions.',
                'AlgT affine gauge is an inverse-depth representative, not the physical body.',
                'Six scene points do not establish foreground surface, new matching, noise robustness or CAD.'])
        write(output/'RESULT.json',result)
        report(output,result)
        return result
    except Exception as exc:
        write(output/'FAILURE.json',dict(status='EXPERIMENT_FAILED',type=type(exc).__name__,reason=str(exc)))
        raise


def report(output,result):
    rows = [('固定共同点 / 训练 / 留出',f"{result['common_count']} / {result['train_count']} / {result['validation_count']}"),
        ('原始像素：留出四维 TLS RMS',result['tls']['raw']['heldout_rms_pixels']),
        ('去畸变像素：留出四维 TLS RMS',result['tls']['undistorted']['heldout_rms_pixels']),
        ('已有标定：Sampson 距离 RMS',result['sampson']['rms']),
        ('校正后两图 y 差 RMS',result['rectified_vertical']['rms']),
        ('修改后六点：实际 AlgT′',result['rectified_exact']['status']),
        ('独立精确证书核验',result['rectified_exact']['independent']['status']),
        ('工程三维候选',result['candidate']['status'])]
    page = '<!doctype html><meta charset="utf-8"><title>Kozlov 真实照片适配实验</title><style>body{font:17px/1.65 system-ui;color:#203246;max-width:1100px;margin:36px auto;padding:0 24px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{width:100%;border-collapse:collapse}aside{background:#fff3d8;padding:16px}a{color:#166b9b}</style>'
    page += '<h1>真实照片 → 校正观测 → Kozlov 构造</h1><aside>这是依赖已有相机标定和对应的六点工程实验。没有生成新物体网格，也没有证明优于 COLMAP。</aside>'
    page += '<p>原始像素完整保留。先去畸变与极线校正，再平均对应点的纵坐标并固定量化，使输入满足精确平行构造的条件。通过 AlgT′ 证书重投影后，以相机基线和视差恢复三维。</p><table>'
    for label,value in rows:
        page += f'<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>'
    page += '</table><p>TLS、Sampson、纵坐标差属于不同残差，不能直接当作算法优劣排名；它们都不是毫米精度。</p>'
    if result['candidate']['status']=='CALIBRATED_SIX_POINT_CANDIDATE':
        page += '<p><a href="calibrated_six_points.ply">新六点 PLY（Blender 可导入，无面）</a> · <a href="old_colmap_six_points.ply">原 COLMAP 同六点 PLY</a> · <a href="POINT_COMPARISON.json">同坐标系逐点对照</a> · <a href="CALIBRATED_POINTS.json">逐点三维与原像素回投影</a></p><p>原方法使用多视图，新方法只取两视图且依赖原相机；差值是方法一致性诊断，不是真值误差或优劣结论。</p>'
    page += '<p><a href="PLAN.json">预冻结方案与划分</a> · <a href="OBSERVATIONS.json">全部坐标变化</a> · <a href="RECTIFIED_SURROGATE.json">六点修改账本</a> · <a href="RECTIFIED_EXACT.json">AlgT′ 精确证书</a> · <a href="RESULT.json">完整结果</a></p>'
    (output/'report.html').write_text(page,encoding='utf-8')
