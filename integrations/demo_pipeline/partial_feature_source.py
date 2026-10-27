"""Read identity-bound native SIFT rows for bounded partial matching diagnostics."""
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from . import parallel_photo_audit as io


MAX_PER_IMAGE = 256


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative plain integer required: ' + label)
    return value


def load_feature_frames(pipeline, frames, masks, refs) -> dict:
    """Return native coordinates and corresponding uint8 SIFT descriptor slices.

    ``pipeline`` is the recorded PIPELINE.json dictionary or its absolute path.
    ``frames`` and ``masks`` are the frozen temporal loader's aligned outputs.
    Point IDs are database image ID + original feature row; no names are guessed.
    """
    local = {}

    def bind(path, declared=None):
        current = io.binding(path)
        if declared:
            if 'path' in declared and Path(declared['path']).resolve() != Path(current['path']):
                raise ValueError('bound path differs')
            if any(current[k] != declared[k] for k in ('bytes', 'sha256') if k in declared):
                raise ValueError('input binding differs')
        for previous in refs:
            if Path(previous['path']).resolve() == Path(current['path']) and any(previous[k] != current[k] for k in ('bytes', 'sha256')):
                raise ValueError('previous input binding differs')
        local[current['path']] = current
        refs.append(current)
        return current

    if not isinstance(pipeline, dict):
        bind(pipeline)
        pipeline = io.read_json(pipeline)
    if len(frames) != len(masks):
        raise ValueError('frame/mask coverage differs')
    sfm_path = Path(pipeline['stages']['sfm']['run_json'])
    if not sfm_path.is_absolute():
        raise ValueError('absolute recorded SfM run path required')
    sfm_path = sfm_path.resolve(strict=True)
    bind(sfm_path)
    sfm = io.read_json(sfm_path)
    root = sfm_path.parent
    fmap_path = root / 'frames_map.jsonl'
    bind(fmap_path)
    rows = [io.loads(s) for s in fmap_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    by_index, by_name = {}, {}
    for row in rows:
        index = _integer(row['input_index'], 'mapped input_index')
        if index in by_index:
            raise ValueError('duplicate mapped input identity')
        by_index[index] = row
        if row.get('staged_image_name') is not None:
            name = row['staged_image_name']
            _integer(row['database_image_id'], 'mapped database_image_id')
            if name in by_name:
                raise ValueError('duplicate mapped image identity')
            if row['upstream_io_run_id'] != sfm['upstream_io_run_id']:
                raise ValueError('mapped upstream identity differs')
            by_name[name] = row
    mask_rows = sfm['effective_options']['foreground']['mask_bindings']
    by_mask = {}
    for row in mask_rows:
        index = _integer(row['input_index'], 'mask input_index')
        if index in by_mask:
            raise ValueError('duplicate mask identity')
        by_mask[index] = row
        mapped = by_index.get(index)
        if mapped is None or mapped.get('staged_image_name') != row['image_name']:
            raise ValueError('mask/map identity differs')
    database = root / 'database.db'
    for suffix in ('-wal', '-journal'):
        sidecar = Path(str(database)+suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError('active SQLite sidecar cannot be bound as a stable database')
    bind(database)
    result_frames = []
    seen = set()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        images = connection.execute('SELECT image_id,name FROM images').fetchall()
        if len({r[0] for r in images}) != len(images) or len({r[1] for r in images}) != len(images):
            raise ValueError('duplicate database image identity')
        for image_id, name in images:
            _integer(image_id, 'database image ID')
            if name not in by_name or by_name[name]['database_image_id'] != image_id:
                raise ValueError('database/map image identity differs')
        db_images = dict(images)
        if set(db_images.values()) != set(by_name):
            raise ValueError('database/map image coverage differs')
        for frame_index, (frame, mask) in enumerate(zip(frames, masks)):
            index = _integer(frame['input_index'], 'frame input_index')
            if index in seen:
                raise ValueError('duplicate current frame identity')
            seen.add(index)
            mapped = by_index.get(index)
            if mapped is None or frame['frame_id'] != mapped['frame_id'] or frame['sha256'] != mapped['canonical_image_sha256']:
                raise ValueError('current frame/map identity differs')
            bind(frame['path'], {'sha256': frame['sha256']})
            record = dict(frame_index=frame_index, input_index=index, db_image_id=None, image_name=None,
                          points=[], descriptors=np.empty((0,128),dtype=np.uint8),
                          selection=dict(original_rows=0, eligible_rows=[], selected_rows=[], omitted_eligible_rows=[], excluded_rows=[],eligible_count=0,selected_count=0,omitted_eligible_count=0,excluded_count=0),
                          status='EXCLUDED_NON_OK_MASK')
            if frame['original_mask_status'] != 'OK':
                if mask is not None or frame.get('mask_binding') is not None or index in by_mask:
                    raise ValueError('non-OK frame has a selected mask binding')
                result_frames.append(record)
                continue
            if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != (frame['height'],frame['width']) or not np.isin(mask,[0,255]).all() or not np.any(mask):
                raise ValueError('invalid bound binary mask array')
            mb = by_mask.get(index)
            if mb is None or frame.get('mask_binding') is None:
                raise ValueError('OK frame missing bound mask')
            if not Path(mb['staged_path']).resolve(strict=True).is_relative_to(root):
                raise ValueError('staged mask escapes SfM run')
            for key in ('source_path','staged_path'):
                bind(mb[key],mb)
            bind(frame['mask_binding']['path'],frame['mask_binding'])
            if Path(mb['source_path']).resolve() != Path(frame['mask_binding']['path']).resolve() or mb['sha256'] != frame['mask_binding']['sha256']:
                raise ValueError('source mask path/binding differs')
            with Image.open(mb['source_path']) as image:
                if image.mode != 'L' or not np.array_equal(np.asarray(image),mask):
                    raise ValueError('supplied mask pixels differ from bound source')
            name = mapped['staged_image_name']
            image_id = mapped['database_image_id']
            staged_image = io.under(root,name)
            bind(staged_image,{'sha256':frame['sha256']})
            with Image.open(staged_image) as image:
                if image.size != (frame['width'],frame['height']):
                    raise ValueError('canonical image dimensions differ')
            kp = connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(image_id,)).fetchall()
            ds = connection.execute('SELECT rows,cols,data,type FROM descriptors WHERE image_id=?',(image_id,)).fetchall()
            if len(kp)!=1 or len(ds)!=1:
                raise ValueError('unique feature matrices required')
            n, cols, raw = kp[0]; dn, dc, desc, dtype = ds[0]
            _integer(n,'feature row count')
            if n>20000 or cols not in (2,4,6) or len(raw or b'')!=n*cols*4:
                raise ValueError('bad keypoint matrix shape/blob')
            if dn!=n or dc!=128 or dtype!=0 or len(desc or b'')!=n*128:
                raise ValueError('bad descriptor matrix shape/blob')
            keys = np.frombuffer(raw or b'',dtype='<f4').reshape(n,cols)
            descriptors = np.frombuffer(desc or b'',dtype=np.uint8).reshape(n,128)
            if not np.isfinite(keys).all():
                raise ValueError('nonfinite keypoint values')
            eligible=[]
            for feature_row,(x,y) in enumerate(keys[:,:2]):
                if 0<=x<frame['width'] and 0<=y<frame['height'] and mask[int(y),int(x)]==255:
                    eligible.append(feature_row)
            selected = eligible if len(eligible)<=MAX_PER_IMAGE else [eligible[i*(len(eligible)-1)//(MAX_PER_IMAGE-1)] for i in range(MAX_PER_IMAGE)]
            chosen=set(selected); allowed=set(eligible)
            record.update(db_image_id=image_id,image_name=name,status='BOUNDED_NATIVE_SIFT_ROWS',
                          points=[dict(id=str(image_id)+':'+str(i),feature_row=i,xy=keys[i,:2].astype(float).tolist()) for i in selected],
                          descriptors=descriptors[selected].copy(),
                          selection=dict(original_rows=n,eligible_rows=eligible,selected_rows=selected,
                              omitted_eligible_rows=[i for i in eligible if i not in chosen],excluded_rows=[i for i in range(n) if i not in allowed],
                              eligible_count=len(eligible),selected_count=len(selected),omitted_eligible_count=len(eligible)-len(selected),excluded_count=n-len(eligible)))
            result_frames.append(record)
    for receipt in local.values():
        if io.binding(receipt['path']) != receipt:
            raise ValueError('input changed during native feature read')
    return dict(schema='bounded-native-feature-frames-v1',frames=result_frames,
                selection=dict(max_points_per_image=MAX_PER_IMAGE,rule='mask-inside rows ascending feature_row; uniform integer positions including first and last when >256',
                    frame_count=len(result_frames),ok_frames=sum(f['status']=='BOUNDED_NATIVE_SIFT_ROWS' for f in result_frames),
                    original_rows=sum(f['selection']['original_rows'] for f in result_frames),
                    eligible_rows=sum(f['selection']['eligible_count'] for f in result_frames),
                    omitted_eligible_rows=sum(f['selection']['omitted_eligible_count'] for f in result_frames),
                    excluded_rows=sum(f['selection']['excluded_count'] for f in result_frames),
                    selected_rows=sum(len(f['points']) for f in result_frames),
                    complete_search=False),
                limitations=['Bounded sampled native SIFT rows, not a complete correspondence search.',
                    'Mask membership is prediction membership, not physical identity or geometry acceptance.',
                    'IDs identify native database image/feature rows, not shared physical points across views.'])
