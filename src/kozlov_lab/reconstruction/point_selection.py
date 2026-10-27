"""Transactional selection of existing native SIFT rows before matching."""
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import struct

from PIL import Image
from kozlov_lab.stereo.point_coverage import greedy_r_cover

MAX_BACKUP_BYTES = 160 * 1024 * 1024


def selection_policy(name, *, profile=None):
    from . import profiles
    if profile is not None: profiles.require_valid_profile(profile)
    if name == 'native':
        return None
    if name != 'rcover-v1':
        raise ValueError('unknown point selection policy: ' + str(name))
    return dict(name=name, policy_id=name, radius_at_1024=16, max_centers=4096,
                max_rows=20000, max_images=192 if profile == profiles.PROFILE_CAD_OBJECT_VIDEO_V1 else 96, coordinates_modified=False,
                minimum_cover_claim=False, order='ascending original row index')


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def _mask(binding):
    for field in ('source_path', 'staged_path'):
        path = Path(binding[field])
        data = path.read_bytes()
        if len(data) != binding['bytes'] or _hash(data) != binding['sha256']:
            raise ValueError('mask binding drift: ' + str(path))
    with Image.open(binding['staged_path']) as image:
        image.load()
        if image.mode != 'L' or not set(image.tobytes()).issubset({0, 255}):
            raise ValueError('mask must be binary L 0/255')
        return image.copy()


def apply_point_selection(database_path, bindings, evidence_path, *, profile=None):
    """Use only a task-owned fresh database; never reuse matched databases.

    Duplicate rows remain separate original identities in the witness ledger;
    selection may choose one representative without merging their descriptors.
    Evidence is staged before commit; VERIFIED is returned only after publication.
    """
    database_path = Path(database_path).resolve(strict=True)
    evidence_path = Path(evidence_path)
    backup = evidence_path.with_name(evidence_path.name + '.before.db')
    pending = evidence_path.with_name(evidence_path.name + '.pending.json')
    failure = evidence_path.with_name(evidence_path.name + '.failed.json')
    verified = evidence_path.with_name(evidence_path.name + '.verified.tmp')
    if any(p.exists() for p in (backup, pending, evidence_path, failure, verified)):
        raise ValueError('fresh evidence paths required')
    if database_path.stat().st_size > MAX_BACKUP_BYTES:
        raise ValueError('database backup exceeds 160 MiB')
    policy = selection_policy('rcover-v1', profile=profile)
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= policy['max_images']:
        raise ValueError('bound mask count exceeds policy capacity')
    connection = sqlite3.connect(database_path)
    committed = False
    try:
        connection.execute('BEGIN IMMEDIATE')
        for table in ('matches', 'two_view_geometries'):
            if connection.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]:
                raise ValueError('matching records already exist: ' + table)
        images = dict(connection.execute('SELECT image_id,name FROM images'))
        names = [b['image_name'] for b in bindings]
        if len(set(names)) != len(names) or set(names) != set(images.values()):
            raise ValueError('image/mask identity coverage differs')
        # Read-only second connection sees the pre-transaction committed snapshot.
        with sqlite3.connect(database_path.resolve().as_uri()+'?mode=ro', uri=True) as source:
            with backup.open('xb'):
                pass
            with sqlite3.connect(backup) as target:
                source.backup(target)
        if backup.stat().st_size > MAX_BACKUP_BYTES:
            raise ValueError('database backup exceeds 160 MiB')
        records = []
        by_name = {b['image_name']: b for b in bindings}
        for ident, name in sorted(images.items()):
            binding = by_name[name]
            mask = _mask(binding)
            camera = connection.execute('SELECT width,height FROM cameras WHERE camera_id=(SELECT camera_id FROM images WHERE image_id=?)', (ident,)).fetchone()
            if camera is None or tuple(camera) != mask.size or max(mask.size)>1024:
                raise ValueError('mask/camera shape differs')
            kp = connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (ident,)).fetchall()
            ds = connection.execute('SELECT rows,cols,data,type FROM descriptors WHERE image_id=?', (ident,)).fetchall()
            if len(kp) != 1 or len(ds) != 1:
                raise ValueError('missing unique feature matrices')
            count, cols, data = kp[0]; dcount, dcols, descriptors, dtype = ds[0]
            data = data or b''; descriptors = descriptors or b''
            if not 0 <= count <= policy['max_rows'] or cols not in (2, 4, 6) or len(data) != count * cols * 4:
                raise ValueError('malformed keypoint matrix')
            if dcount != count or dcols != 128 or dtype != 0 or len(descriptors) != count * 128:
                raise ValueError('descriptor rows/type/bytes differ')
            rows = [struct.unpack_from('<'+'f'*cols, data, i*cols*4) for i in range(count)]
            if any(not all(math.isfinite(v) for v in row) for row in rows):
                raise ValueError('nonfinite keypoint row')
            coordinates = [[r[0], r[1]] for r in rows]
            pixels = mask.load(); width, height = mask.size
            inside = [i for i, (x, y) in enumerate(coordinates)
                      if 0 <= x < width and 0 <= y < height and pixels[int(x), int(y)] == 255]
            cover = greedy_r_cover([coordinates[i] for i in inside],
                                   16 * max(mask.size) / 1024, max_centers=4096)
            if not cover['complete']:
                raise ValueError('r-cover budget incomplete')
            selected = sorted(inside[i] for i in cover['selected_indices'])
            new_keys = b''.join(data[i*cols*4:(i+1)*cols*4] for i in selected)
            new_desc = b''.join(descriptors[i*128:(i+1)*128] for i in selected)
            connection.execute('UPDATE keypoints SET rows=?,data=? WHERE image_id=?', (len(selected), new_keys, ident))
            connection.execute('UPDATE descriptors SET rows=?,data=? WHERE image_id=?', (len(selected), new_desc, ident))
            if connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (ident,)).fetchone() != (len(selected), cols, new_keys):
                raise ValueError('keypoint byte readback failed')
            if connection.execute('SELECT rows,cols,data,type FROM descriptors WHERE image_id=?', (ident,)).fetchone() != (len(selected), 128, new_desc, 0):
                raise ValueError('descriptor byte readback failed')
            mapping = [None] * count
            for new, old in enumerate(selected): mapping[old] = new
            records.append(dict(image_id=ident, image_name=name, mask_sha256=binding['sha256'],
                original_coordinates=coordinates, original_rows=count, retained_rows=len(selected),
                eligible_original_rows=inside, selected_original_rows=selected,
                original_to_new=mapping, cover=cover,
                witness_original_rows=[inside[i] if i >= 0 else None for i in cover['nearest_selected_indices']],
                before_keypoints_sha256=_hash(data), before_descriptors_sha256=_hash(descriptors),
                after_keypoints_sha256=_hash(new_keys), after_descriptors_sha256=_hash(new_desc)))
        # Recheck both source and staged masks immediately before committing.
        for binding in bindings: _mask(binding)
        audit = dict(status='VERIFIED', policy_id='rcover-v1', policy=policy, images=records, per_image=records,
                     checked_images=len(records), original_rows=sum(r['original_rows'] for r in records),
                     retained_rows=sum(r['retained_rows'] for r in records),
                     before_database_path=str(backup.resolve()), before_database_sha256=_hash(backup.read_bytes()),
                     before_total=sum(r['original_rows'] for r in records), after_total=sum(r['retained_rows'] for r in records),
                     source_database_backup=str(backup.resolve()), descriptors_preserved=True,
                     evidence_path=str(evidence_path.resolve()), coordinates_modified=False,
                     alias_rule='Duplicate source rows retain separate witness identities; only selected original descriptor rows survive.')
        _save(pending, dict(audit, status='PREPARED_NOT_COMMITTED'))
        # No VERIFIED file exists until commit has completed. Publication failures
        # after commit raise with a FAILED sidecar and the full before backup.
        _save(evidence_path, dict(audit, status='PREPARED_NOT_COMMITTED'))
        connection.commit(); committed = True
        _save(verified, audit)
        os.replace(verified, evidence_path)
        return audit
    except Exception as exc:
        if not committed: connection.rollback()
        try:
            _save(failure, dict(status='FAILED', database_committed=committed, error=str(exc)))
        except OSError:
            pass
        raise
    finally:
        connection.close()
