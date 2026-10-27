"""Explicit source choice; never changes source states or creates mask pixels."""
from pathlib import Path

from . import manifest_io as mio


def select_sources(path, manifests, index, *, derived_model=None):
    ledger_path = Path(path)
    if not ledger_path.is_absolute():
        raise mio.InputError("selection ledger path must be absolute")
    ledger_binding = mio.file_record(ledger_path)
    if ledger_binding["bytes"] > 4 * 1024**2:
        raise mio.InputError("selection ledger exceeds 4 MiB")
    doc = mio.load_json_strict(ledger_path, "selection ledger")
    if mio.file_record(ledger_path) != ledger_binding:
        raise mio.InputError("selection ledger changed during read")
    normal_fields = {"schema_version", "manifests", "selections"}
    subset_mode = isinstance(doc, dict) and 'subset_certificate' in doc
    fields = normal_fields | {'subset_certificate','excluded_source_rows'} if subset_mode else normal_fields
    if not isinstance(doc, dict) or set(doc) != fields:
        raise mio.InputError("selection ledger requires exactly schema_version/manifests/selections")
    if doc["schema_version"] != "object-mask-selection-v1":
        raise mio.InputError("unknown selection ledger schema")
    sources = {}
    for manifest in manifests:
        key = mio.normalised(manifest["manifest_path"])
        if key in sources:
            raise mio.InputError("duplicate supplied manifest path")
        sources[key] = manifest
    declared = doc["manifests"]
    if not isinstance(declared, list) or len(declared) != len(sources):
        raise mio.InputError("ledger must bind every supplied manifest exactly once")
    declared_keys = set()
    for item in declared:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise mio.InputError("ledger manifest binding requires path/sha256")
        value = mio.require_str(item["path"], "ledger manifest path")
        if not Path(value).is_absolute():
            raise mio.InputError("ledger manifest path must be absolute")
        key = mio.normalised(value)
        if key in declared_keys or key not in sources:
            raise mio.InputError("duplicate or unknown ledger manifest")
        declared_keys.add(key)
        digest = mio.require_sha256(item["sha256"], "ledger manifest sha256")
        actual = mio.file_record(value)
        if digest != sources[key]["manifest_sha256"] or digest != actual["sha256"]:
            raise mio.InputError("ledger source manifest hash mismatch")
    if len({m["object_id"] for m in manifests}) != 1:
        raise mio.InputError("ledger sources have different object_id values")
    if any(m["source_io"] != manifests[0]["source_io"] for m in manifests):
        raise mio.InputError("ledger sources have different T01 source_io identities")
    targets = {(v["input_index"], v["frame_id"]) for v in index["by_name"].values()}
    rows, bindings = {}, [ledger_binding, mio.file_record(__file__)]
    subset = None
    general = False
    excluded_identities = set()
    if subset_mode:
        certificate_doc = mio.load_json_strict(Path(doc['subset_certificate']['path']), 'subset certificate')
        general = certificate_doc.get('schema_version') == 'native-general-subset-v1'
        if general:
            from mvs_surface.general_subset import validate_reference
        else:
            from mvs_surface.subset import validate_reference
        if derived_model is None:
            raise mio.InputError('subset selection requires the actual bridge source model')
        subset = validate_reference(doc['subset_certificate'],derived_model=derived_model,
                                    sfm_run=index['sfm_root'],io_run=index['t01_root'])
        expected_targets={(r['input_index'],r['frame_id']) for r in subset['selected_inputs']}
        if targets != expected_targets:
            raise mio.InputError('bridge targets differ from certified43 subset')
        if general:
            excluded_identities = {(r['input_index'], r['frame_id']) for r in subset['excluded_inputs']}
        else:
            excluded_identity=(subset['excluded']['input_index'],subset['excluded']['frame_id'])
            excluded_identities = {excluded_identity}
        bindings.extend(subset['bindings'])
    outside = []
    for key, manifest in sources.items():
        for frame in manifest["frames"]:
            identity = (frame["input_index"], frame["frame_id"])
            if identity not in targets:
                if (subset is None or identity not in excluded_identities or
                        (not general and (frame['status'] != 'NO_MATCH'
                         or frame['candidates'] or frame['chosen'] is not None))):
                    raise mio.InputError("ledger source frame outside selected bridge set or unknown identity")
                # Full canonical/T01 checks also run through bind_frames on the
                # original source manifest in CLI preflight, before publishing.
                excluded = None if general else subset['excluded']
                if not general and (not isinstance(frame['source_image'],str) or
                        frame['source_image_sha256'] != excluded['canonical']['sha256'] or
                        mio.normalised(frame['source_image']) != mio.normalised(excluded['canonical']['path']) or
                        (frame['width'],frame['height']) != (excluded['width'],excluded['height'])):
                    raise mio.InputError('excluded59 canonical identity differs from certificate')
                outside.append({'manifest_path':manifest['manifest_path'],
                    'manifest_sha256':manifest['manifest_sha256'],'input_index':identity[0],
                    'frame_id':identity[1],'reason':'CERTIFIED_VIEW_EXCLUDED' if general else 'NO_MATCH_VIEW_EXCLUDED'})
            if not isinstance(frame["source_image"], str) or not frame["source_image"]:
                raise mio.InputError("ledger source row lacks canonical source_image identity")
            row_key = (key, identity)
            if row_key in rows:
                raise mio.InputError("duplicate identity within source manifest")
            rows[row_key] = frame
            # Bind and validate even candidate files of unchosen source rows.
            for candidate in frame["candidates"]:
                _, record = mio.read_label_png(candidate["resolved_mask_path"], frame["width"], frame["height"])
                if (record["sha256"] != candidate["mask_sha256"] or
                        candidate["foreground_pixels"] is not None and
                        record["foreground_pixels"] != candidate["foreground_pixels"]):
                    raise mio.InputError("ledger source candidate bytes/pixel declaration mismatch")
                bindings.append(mio.file_record(candidate["resolved_mask_path"]))
    if subset is not None:
        declared_outside=doc['excluded_source_rows']
        if (not isinstance(declared_outside,list) or (not general and len(outside)!=2) or
                len(declared_outside)!=len(outside) or
                any(not isinstance(v,dict) or set(v)!={'manifest_path','manifest_sha256','input_index','frame_id','reason'} or type(v.get('input_index')) is not int for v in declared_outside)):
            raise mio.InputError('subset ledger must explicitly account for both original NO_MATCH59 rows')
        for item in declared_outside:
            for key in ('manifest_path','frame_id','reason'):
                mio.require_str(item[key],'excluded '+key)
            mio.require_sha256(item['manifest_sha256'],'excluded manifest hash')
            if not Path(item['manifest_path']).is_absolute():
                raise mio.InputError('excluded manifest path must be absolute')
        normalise=lambda v:(mio.normalised(v['manifest_path']),v['manifest_sha256'],v['input_index'],v['frame_id'],v['reason'])
        if sorted(map(normalise,declared_outside)) != sorted(map(normalise,outside)):
            raise mio.InputError('excluded source declarations differ from actual59 rows')
    selections = doc["selections"]
    if not isinstance(selections, list):
        raise mio.InputError("ledger selections must be a list")
    chosen = {}
    for item in selections:
        fields = {"input_index", "frame_id", "manifest_path", "selected_candidate_id", "selected_mask_sha256"}
        if not isinstance(item, dict) or set(item) != fields:
            raise mio.InputError("ledger selection fields differ from contract")
        identity = (mio.require_int(item["input_index"], "ledger input_index"),
                    mio.require_str(item["frame_id"], "ledger frame_id"))
        if identity not in targets or identity in chosen:
            raise mio.InputError("extra or duplicate ledger target identity")
        source_path = mio.require_str(item["manifest_path"], "ledger selected manifest")
        if not Path(source_path).is_absolute():
            raise mio.InputError("ledger selected manifest path must be absolute")
        source_key = mio.normalised(source_path)
        frame = rows.get((source_key, identity))
        if frame is None or frame["status"] != "OK" or frame["chosen"] is None:
            raise mio.InputError("ledger must select an existing OK source row")
        if (mio.require_str(item["selected_candidate_id"], "ledger candidate") != frame["selected_candidate_id"] or
                mio.require_sha256(item["selected_mask_sha256"], "ledger selected sha256") != frame["selected_mask"]["sha256"]):
            raise mio.InputError("ledger selected candidate/hash differs from source selection")
        chosen[identity] = source_key
    if set(chosen) != targets:
        raise mio.InputError("ledger omits selected bridge targets")
    effective, excluded = [], []
    for source_key, manifest in sources.items():
        kept = []
        for frame in manifest["frames"]:
            identity = (frame["input_index"], frame["frame_id"])
            if subset is not None and identity in excluded_identities:
                excluded.append({'input_index':identity[0],'frame_id':identity[1],
                    'original_status':frame['status'],'source_manifest':manifest['manifest_path'],
                    'source_manifest_sha256':manifest['manifest_sha256'],
                    'reason':'explicit certified view exclusion; no replacement mask' if general else 'explicit certified view59 exclusion; no replacement mask',
                    'subset_certificate':subset['certificate'],'replacement_manifest':None})
                continue
            if chosen[identity] == source_key:
                kept.append(frame)
            else:
                replacement = sources[chosen[identity]]
                selected = rows[(chosen[identity], identity)]
                excluded.append({"input_index": identity[0], "frame_id": identity[1],
                    "original_status": frame["status"], "source_manifest": manifest["manifest_path"],
                    "source_manifest_sha256": manifest["manifest_sha256"],
                    "reason": "explicit ledger selected another source for this target",
                    "replacement_manifest": replacement["manifest_path"],
                    "replacement_manifest_sha256": replacement["manifest_sha256"],
                    "replacement_candidate_id": selected["selected_candidate_id"],
                    "replacement_mask_sha256": selected["selected_mask"]["sha256"]})
        effective.append(dict(manifest, frames=kept))
    return {"effective_manifests": effective, "bindings": bindings,
            "certified_excluded": excluded_identities if general else set(),
            "record": {"schema_version": doc["schema_version"], "ledger": ledger_binding,
                       "selections": selections, "unselected_source_rows": excluded,
                       "source_runs": [{"path": m["manifest_path"], "sha256": m["manifest_sha256"],
                                        "status": m["status"], "counts": m["counts"]} for m in manifests],
                       "quality_status": "NOT_EVALUATED", "source_states_modified": False,
                       "implementation": mio.file_record(__file__)}}
