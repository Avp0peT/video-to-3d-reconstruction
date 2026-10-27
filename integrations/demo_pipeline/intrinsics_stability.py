"""Read-only focal sensitivity and preprocessing evidence; never calibrates a camera.

F storage: https://colmap.github.io/database.html (row-major float64).
Essential constraint: HZ, section 9.6, Result 9.17. The input F was fitted on
these same matches. Its necessary consistency is not independent validation.
"""
from collections import Counter
from pathlib import Path
import html
import json
import math
import sqlite3
import struct

import numpy as np

from . import parallel_photo_audit as native
from .focal_profile import profile_fundamental


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite JSON value')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    if isinstance(value, list):
        for item in value:
            _finite(item)
    return value


class Inputs:
    def __init__(self):
        self.refs = {}

    def bind(self, path):
        ref = native.binding(path)
        previous = self.refs.setdefault(ref['path'], ref)
        if previous != ref:
            raise ValueError('input drift: '+ref['path'])
        return ref

    def read(self, path):
        self.bind(path)
        return _finite(native.read_json(path))

    def rows(self, path):
        self.bind(path)
        rows = [_finite(native.loads(line)) for line in Path(path).read_text(
            encoding='utf-8-sig').splitlines() if line.strip()]
        if len(rows) > 10000 or any(type(r) is not dict for r in rows):
            raise ValueError('bounded object rows required')
        return rows

    def verify(self):
        for ref in self.refs.values():
            if native.binding(ref['path']) != ref:
                raise ValueError('input drift: '+ref['path'])


def _unique(rows, key):
    out = {}
    for row in rows:
        value = row[key]
        if type(value) is not int or value < 0 or value in out:
            raise ValueError('unique nonnegative integer '+key+' required')
        out[value] = row
    return out


def _argument(argv, name):
    if type(argv) is not list or argv.count(name) != 1:
        raise ValueError('one recorded argument required: '+name)
    index = argv.index(name)+1
    if index >= len(argv) or type(argv[index]) is not str:
        raise ValueError('missing recorded argument: '+name)
    return argv[index]


def preprocessing(pipeline, run_path, run, frames, inputs):
    result = dict(status='UNKNOWN', optical_zoom='UNKNOWN', physical_intrinsics='UNKNOWN',
                  reason='No bound preprocessing provenance supplied', calibration_accepted=False)
    if pipeline is None:
        return result
    pipeline = Path(pipeline).resolve(strict=True)
    p = inputs.read(pipeline/'pipeline.json')
    sfm = p['stages']['sfm']
    if Path(sfm['run_json']).resolve(strict=True) != run_path:
        raise ValueError('pipeline refers to a different SfM run')
    io_root = Path(_argument(sfm['argv'], '--io-run')).resolve(strict=True)
    if io_root.is_file():
        io_root = io_root.parent
    io_path = io_root/'run.json'
    io = inputs.read(io_path)
    if io.get('contract_version') != 'photo-io-v1' or io.get('run_id') != run['upstream_io_run_id']:
        raise ValueError('T01/SfM run identity differs')
    rows = _unique(inputs.rows(io_root/'frames.jsonl'), 'input_index')
    transforms, sizes = [], set()
    for f in frames.values():
        row = rows[f['input_index']]
        if row['frame_id'] != f['frame_id'] or row['status'] != 'OK':
            raise ValueError('T01/frame identity differs')
        image = inputs.bind(native.under(io_root, row['image_output']))
        if image['sha256'] != f['canonical_image_sha256']:
            raise ValueError('T01 canonical bytes differ from staged image')
        size = [row['canonical_width'], row['canonical_height']]
        if size != f['checked_size']:
            raise ValueError('T01 canonical dimensions differ')
        transforms.append(row['raw_to_canonical'])
        sizes.add(tuple(size))
    identity = all(t == [[1,0,0],[0,1,0],[0,0,1]] for t in transforms)
    result.update(status='PHOTO_IO_BOUND', reason=None, canonical_sizes=sorted(sizes),
                  raw_to_canonical_identity=identity, image_count=len(frames))
    if p.get('source_kind') != 'video':
        return result
    # The recorded prepare_object output and video-photo contract locate T01;
    # neither a run-id directory search nor a filename-derived frame identity is used.
    video_root = Path(_argument(p['stages']['input']['argv'], '--output')).resolve(strict=True)/'video'
    v = inputs.read(video_root/'RESULT.json')
    if v.get('contract_version') != 'video-photo-v1' or v.get('status') != 'SUCCESS':
        raise ValueError('successful video-photo provenance required')
    if (Path(v['photo_io']['run']).resolve(strict=True) != io_path
            or inputs.bind(io_path)['sha256'] != v['photo_io']['sha256']
            or inputs.bind(io_root/'frames.jsonl')['sha256'] != v['photo_io']['frames_sha256']):
        raise ValueError('video/T01 binding differs')
    source = Path(v['source_video']).resolve(strict=True)
    if source != Path(p['source']).resolve(strict=True):
        raise ValueError('pipeline/video source differs')
    ref = inputs.bind(source)
    if not ref['sha256'] == v['source_sha256'] == v['source_sha256_after']:
        raise ValueError('video source bytes differ')
    timeline = inputs.read(video_root/'timeline.json')
    extraction = inputs.read(video_root/'extract.process.json')
    if {k:value for k,value in timeline.items() if k != 'frames'} != v.get('timeline'):
        raise ValueError('video timeline metadata differs from committed RESULT')
    if Path(_argument(extraction['argv'], '-i')).resolve(strict=True) != source:
        raise ValueError('extraction command refers to different source video')
    filter_text = _argument(extraction['argv'], '-vf')
    selected = _unique([r for r in v['frames'] if r.get('status') == 'PHOTO_IO_OK'], 'photo_input_index')
    ratios = set()
    for f in frames.values():
        r = selected[f['input_index']]
        if r['photo_frame_id'] != f['frame_id'] or [r['width'],r['height']] != f['checked_size']:
            raise ValueError('video frame identity/dimensions differ')
        ratios.add((r['width']/timeline['width'], r['height']/timeline['height']))
    result.update(status='VIDEO_PREPROCESSING_BOUND', source_video=ref,
        source_dimensions=[timeline['width'],timeline['height']],
        sample_aspect_ratio=timeline.get('sample_aspect_ratio'),
        rotation_degrees=timeline.get('rotation_degrees'),
        display_matrix=timeline.get('display_matrix'), recorded_extraction_filter=filter_text,
        output_to_decoded_size_ratios=sorted(ratios),
        equal_axis_size_ratios=all(abs(x-y) < 1e-12 for x,y in ratios),
        crop_filter_recorded=('crop=' in filter_text),
        optical_zoom='UNKNOWN', reason='Pixel dimensions and scaling do not establish optical focal stability')
    # Optional known sidecar is admitted only after binding both named video files.
    comparison_path = source.parent/'COMPARISON.json'
    if comparison_path.is_file():
        comp = inputs.read(comparison_path)
        if Path(comp['derived']['path']).resolve(strict=True) != source or comp['derived']['sha256'] != ref['sha256']:
            raise ValueError('normalization sidecar refers to different video')
        original = inputs.bind(comp['original']['path'])
        if original['sha256'] != comp['original']['sha256']:
            raise ValueError('normalization original video drift')
        result['normalization'] = dict(status=comp['status'], assumption=comp.get('assumption'),
            original=original, decoded_comparison_record=comp.get('all_decoded_frame_md5_pts_dts_duration_size_equal'),
            evidence='Previously recorded decoded comparison; this diagnostic does not decode video again')
    return result


def _png_size(path):
    with open(path, 'rb') as f:
        header = f.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('staged PNG IHDR required')
    return list(struct.unpack('>II', header[16:24]))


def _matrix(blob):
    if blob is None:
        return None
    if len(blob) != 72:
        raise ValueError('3x3 float64 matrix requires 72 bytes')
    value = np.frombuffer(blob, dtype='<f8').reshape(3,3)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite stored matrix')
    return value


def _sampson(F, a, b):
    x = np.column_stack((a,np.ones(len(a))))
    y = np.column_stack((b,np.ones(len(b))))
    fx, fty = x@F.T, y@F
    denom = np.sum(fx[:,:2]**2,axis=1)+np.sum(fty[:,:2]**2,axis=1)
    valid = denom > 0
    d = np.abs(np.sum(y*fx,axis=1)[valid])/np.sqrt(denom[valid])
    return dict(count=int(len(d)), undefined_count=int((~valid).sum()),
                median_pixels=float(np.median(d)) if len(d) else None,
                max_pixels=float(np.max(d)) if len(d) else None,
                independent_validation=False, source='same stored inliers used by geometry estimation')


def audit_intrinsics(sfm_run, output, pipeline=None):
    root = Path(sfm_run).resolve(strict=True)
    run_path = root if root.is_file() else root/'run.json'
    root = run_path.parent
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('fresh output outside SfM source required')
    if pipeline is not None:
        p = Path(pipeline).resolve(strict=True)
        if output.is_relative_to(p/'input') or p.is_relative_to(output):
            raise ValueError('output overlaps preprocessing source')
    inputs = Inputs()
    run = inputs.read(run_path)
    if run.get('contract_version') not in ('sfm-native-object-v1','sfm-native-v1','sfm-native-profile-v1'):
        raise ValueError('unsupported SfM contract')
    fmap = _unique(inputs.rows(root/'frames_map.jsonl'), 'input_index')
    mapped = _unique([r for r in fmap.values() if r.get('database_image_id') is not None], 'database_image_id')
    if not 2 <= len(mapped) <= 96:
        raise ValueError('2..96 database images required')
    database = root/'database.db'
    inputs.bind(database)
    def reject_journal():
        for suffix in ('-wal','-journal'):
            side = Path(str(database)+suffix)
            if side.exists() and side.stat().st_size:
                raise ValueError('active database journal is not a frozen input')
    reject_journal()
    cameras, keypoints, pairs = {}, {}, []
    # immutable prevents SQLite sidecar creation; only admitted for a frozen,
    # journal-free DB and rechecked hashes at completion.
    db = sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        image_rows = db.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id').fetchall()
        if {r[0] for r in image_rows} != set(mapped):
            raise ValueError('database/frame-map image set differs')
        for cid, model, width, height, params, prior in db.execute('SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras'):
            if len(cameras) >= 96 or len(params) > 128 or len(params)%8 or not 1 <= width <= 16384 or not 1 <= height <= 16384:
                raise ValueError('bounded camera representation required')
            a = np.frombuffer(params,dtype='<f8')
            if not np.isfinite(a).all():
                raise ValueError('nonfinite camera')
            cameras[cid] = dict(camera_id=cid,model=model,width=width,height=height,
                                params=a.tolist(),prior_focal_length=prior,
                                physically_calibrated=False)
        for iid, name, cid in image_rows:
            row = mapped[iid]
            if row['staged_image_name'] != name or row['upstream_io_run_id'] != run['upstream_io_run_id']:
                raise ValueError('database/frame-map identity differs')
            image = native.under(root,name)
            ref = inputs.bind(image)
            if ref['sha256'] != row['canonical_image_sha256']:
                raise ValueError('staged image bytes differ')
            c = cameras[cid]
            row = dict(row, camera_id=cid, checked_size=_png_size(image))
            mapped[iid] = row
            if row['checked_size'] != [c['width'],c['height']]:
                raise ValueError('camera/staged image size differs')
            r = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
            if not r or not 0 <= r[0] <= 20000 or r[1] not in (2,4,6) or len(r[2]) != r[0]*r[1]*4:
                raise ValueError('bounded keypoints required')
            xy = np.frombuffer(r[2],dtype='<f4').reshape(r[0],r[1])[:,:2]
            if not np.isfinite(xy).all():
                raise ValueError('nonfinite keypoint')
            keypoints[iid] = xy
        rows = db.execute('SELECT pair_id,rows,cols,data,config,F,H FROM two_view_geometries ORDER BY pair_id').fetchmany(4609)
        if len(rows) > 4608:
            raise ValueError('pair count exceeds bound')
        for pid, n, cols, blob, config, fb, hb in rows:
            a,b = divmod(pid,2147483647)
            if not a < b or a not in mapped or b not in mapped:
                raise ValueError('pair identity differs')
            if not 0 <= n <= 100000 or cols != 2 or len(blob or b'') != n*8:
                raise ValueError('invalid stored match matrix')
            F, H = _matrix(fb), _matrix(hb)
            p = dict(pair_id=pid,image_ids=[a,b],input_indices=[mapped[a]['input_index'],mapped[b]['input_index']],
                     config=config,inlier_count=n,has_fundamental=F is not None,has_homography=H is not None,
                     degeneracy_status='NOT_EVALUATED_FROM_F_ALONE')
            ca,cb = cameras[mapped[a]['camera_id']], cameras[mapped[b]['camera_id']]
            reason = None
            if n == 0:
                reason = 'NO_VERIFIED_INLIERS'
            elif config not in (2,3):
                reason = 'AMBIGUOUS_GEOMETRY_CONFIGURATION'
            elif F is None:
                reason = 'NO_STORED_F'
            elif ca != cb:
                reason = 'SHARED_CAMERA_PROFILE_NOT_APPLICABLE'
            elif ca['model'] != 2 or len(ca['params']) != 4 or ca['params'][0] <= 0 or ca['params'][3] != 0:
                reason = 'ZERO_DISTORTION_SIMPLE_RADIAL_PROFILE_NOT_APPLICABLE'
            if reason:
                p.update(status='REFUSED',reason=reason)
            else:
                matches = np.frombuffer(blob,dtype='<u4').reshape(n,2)
                if (matches[:,0] >= len(keypoints[a])).any() or (matches[:,1] >= len(keypoints[b])).any():
                    raise ValueError('match keypoint identity out of range')
                p['sampson'] = _sampson(F,keypoints[a][matches[:,0]],keypoints[b][matches[:,1]])
                profile = profile_fundamental(F,ca['params'][1:3],cb['params'][1:3],max(ca['width'],ca['height']),config=config)
                p.update(status=profile['status'],profile=profile,fundamental=F.tolist())
            pairs.append(p)
    finally:
        db.close()
    provenance = preprocessing(pipeline,run_path,run,mapped,inputs)
    models = []
    for model in run.get('models',[]):
        path = native.under(root,model['binary_path'])/'cameras.bin'
        inputs.bind(path)
        reader = native.Reader(path)
        count, = reader.take('<Q')
        if not 1 <= count <= 96:
            raise ValueError('bounded model camera count required')
        for _ in range(count):
            cid,kind,w,h = reader.take('<iiQQ')
            if kind != 2:
                raise ValueError('model summary supports SIMPLE_RADIAL only')
            params = list(reader.take('<dddd'))
            if not all(math.isfinite(x) for x in params):
                raise ValueError('nonfinite fitted camera')
            models.append(dict(model_id=model['model_id'],camera_id=cid,width=w,height=h,params=params,
                               evidence='fitted estimate in separate reconstructed component, not an optical measurement'))
        reader.finish()
    reject_journal()
    inputs.verify()
    evaluated = [p for p in pairs if p['status']=='FOCAL_PROFILE_EVALUATED']
    result = dict(schema='intrinsics-stability-diagnostic-v1',status='INTRINSICS_DIAGNOSTIC_COMPLETE',
        sfm_run=str(run_path),source_unchanged=True,preprocessing=provenance,
        initial_cameras=list(cameras.values()),effective_options=run.get('effective_options'),
        fitted_component_cameras=models,pairs=pairs,
        counts=dict(images=len(mapped),stored_pairs=len(pairs),profiled_pairs=len(evaluated),
                    configurations=dict(Counter(str(p['config']) for p in pairs)),
                    profile_states=dict(Counter(p['profile']['focal_observability_status'] for p in evaluated))),
        decision=dict(status='FOCAL_OBSERVABILITY_NOT_ESTABLISHED',calibration_accepted=False,
                      recommended_camera_params=None,new_ba_authorized=False,
                      reasons=['Optical zoom is not measured by image dimensions or a shared database camera ID',
                               'Zero distortion and fixed principal point are unverified physical assumptions',
                               'A minimum of essential consistency alone does not rule out planar/rotation degeneracy',
                               'The same stored matches estimated F; no independent geometric validation']),
        native_model_executions=0,geometry_changed=False,scale_status='unknown',
        source_bindings=list(inputs.refs.values()),
        implementation=[native.binding(Path(__file__)),native.binding(Path(__file__).with_name('focal_profile.py'))])
    output.mkdir(parents=True,exist_ok=False)
    native.write_json(output/'INTRINSICS.json',result)
    render(output,result)
    return result


def render(output,result):
    esc=lambda value:html.escape(str(value))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>焦距与内参稳定性检查</title>
<style>body{font:16px/1.65 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#233348}aside{background:#fff0c9;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}svg{width:100%;max-width:850px;background:#f3f6fa}select{font:inherit;padding:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>焦距与内参稳定性检查</h1><aside>只读诊断完成；真实焦距尚未确定。没有修改原对应、相机或模型，也没有启动新的 BA。</aside>
<p>相同物理点在图像中的位置可随姿态、焦距、裁剪和缩放改变。此处检查已有 F 是否能约束共享焦距；数据库共享相机与 k=0 是建模假设，不能证明物理标定。</p>'''
    page+='<h2>输入与预处理</h2><pre>'+esc(json.dumps(result['preprocessing'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>逐图对的条件曲线</h2><p>横轴为 f / 图像长边（对数轴，0.1–10，共101点）；纵轴为原始 E=K₂ᵀFK₁ 到本质矩阵奇异值条件的相对平方距离。越低仅表示此条件更接近，不是像素误差、置信度或三维精度。未投影 E 后再评分。</p><select id="pairs"></select><p id="info"></p><svg viewBox="0 0 850 280"><path id="curve" fill="none" stroke="#147c9e" stroke-width="2"/><path d="M40 20 V250 H825" fill="none" stroke="#8190a0"/></svg>'
    page+='<table><tr><th>原始帧对</th><th>内点</th><th>状态</th><th>网格最低 f(px)</th><th>qess</th></tr>'
    for p in result['pairs']:
        if not p['inlier_count']:
            continue
        pr=p.get('profile',{})
        curve=pr.get('curve',[])
        best=curve[pr['minimum_grid_index']] if curve else None
        page+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in (p['input_indices'],p['inlier_count'],pr.get('focal_observability_status',p.get('reason',pr.get('reason_code'))),f"{best['focal']:.2f}" if best else '—',f"{best['qess']:.4g}" if best else '—'))+'</tr>'
    page+='</table><p>边界最低、平坦或多极小均按有限网格定义；内点最低也不能排除平面、纯旋转、错误主点或畸变影响。图对共享图像，不是独立样本，不能平均后称为标定。</p>'
    page+='<h2>分量内拟合的相机估计</h2><pre>'+esc(json.dumps(result['fitted_component_cameras'],ensure_ascii=False,indent=2))+'</pre><p>这些是不同重建分量的估计，不能据此认定拍摄时发生变焦。</p><p><a href="INTRINSICS.json">全部101点曲线、来源哈希与判定</a> · <a href="https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf">HZ §9.6 原理</a> · <a href="https://colmap.github.io/database.html">COLMAP 数据格式</a></p>'
    data=json.dumps([p for p in result['pairs'] if p.get('profile',{}).get('curve')],ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page+='<script>const data='+data+''';const sel=document.getElementById('pairs');data.forEach((p,i)=>{let o=document.createElement('option');o.value=i;o.textContent=p.input_indices.join(' – ');sel.append(o)});function draw(){if(!data.length)return;let p=data[+sel.value],c=p.profile.curve,m=Math.max(...c.map(r=>r.qess),1e-15);document.getElementById('curve').setAttribute('d',c.map((r,i)=>(i?'L':'M')+(40+i*7.85)+' '+(250-r.qess/m*220)).join(' '));document.getElementById('info').textContent=p.profile.focal_observability_status+'；纵轴上限 '+m.toPrecision(4)+'；退化未排除'}sel.onchange=draw;draw();</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
