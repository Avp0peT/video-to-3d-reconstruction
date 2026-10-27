"""Versioned broad SAME-ID C/J candidates; fixed native calibration, no old XYZ fit."""
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import time
import numpy as np

from . import multi_pair_ablation as previous
from . import heldout_multiview as heldout
from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from kozlov_lab.stereo.baseline import rectified_parallel as rect
from kozlov_lab.stereo.baseline import multiview_fixed as fixed

PROFILE = dict(name='expanded-same-id-c-j-v1', max_pairs=24, max_groups=1600,
               soft_seconds=300, grid_bits=18, common_y=True, normalized_scale=1,
               max_iterations=20, no_regrouping=True)


def assign(points):
    """Greedy maximum uncovered singleton-track coverage, ties ascending pair IDs.

    Every chosen pair owns all its uncovered tracks, including its terminal tail.
    Failure cannot move an owned track to another pair.
    """
    eligible = defaultdict(set)
    for pid, point in sorted(points.items()):
        for pair in combinations(sorted(point['track']), 2):
            eligible[pair].add(pid)
    uncovered = set(points); selected = []; groups = 0
    ranking = [dict(image_pair=list(pair), shared_count=len(ids))
               for pair, ids in sorted(eligible.items(), key=lambda v: (-len(v[1]), v[0]))]
    while len(selected) < PROFILE['max_pairs']:
        choices = [(pair, sorted(ids & uncovered)) for pair, ids in eligible.items()]
        choices = [v for v in choices if len(v[1]) >= 6]
        if not choices: break
        pair, ids = min(choices, key=lambda v: (-len(v[1]), v[0]))
        blocks = [dict(block_index=i, point_ids=ids[i*6:i*6+6],
                       scheduled=groups+i < PROFILE['max_groups']) for i in range(len(ids)//6)]
        groups += len(blocks)
        selected.append(dict(image_pair=list(pair), point_ids=ids, groups=blocks,
                             tail_point_ids=ids[len(blocks)*6:]))
        uncovered.difference_update(ids)
    return dict(selected=selected, pair_ranking=ranking, unassigned_point_ids=sorted(uncovered),
                rule='maximum uncovered singleton shared tracks; tie ascending image IDs; ascending point IDs in consecutive six')


def generate(prior_input, output):
    old, frozen, names, images, points, reference = previous._load(prior_input)
    output = Path(output).resolve(); root = names['run.json'].parent
    if output.exists(): raise ValueError('new exclusive output required')
    if output.is_relative_to(root) or output.is_relative_to(old.parent):
        raise ValueError('output cannot be inside immutable source')
    selection = assign(points)
    cameras, poses = bridge.read_calibration(names['cameras.bin'], names['images.bin'])
    # Bind every singleton view used by J, even if absent from the C pairs.
    frame_pairs = [dict(image_pair=[i]) for i in sorted(images)]
    frames = previous._bind_photos(root, native.read_json(names['run.json']), images, frame_pairs, names)
    refs = list(frozen['source_bindings']) + [native.binding(old)]
    known = {r['path'] for r in refs}
    refs += [f['binding'] for f in frames if f['binding']['path'] not in known]
    for ref in refs: bridge.verify_binding(ref)
    output.mkdir(parents=True, exist_ok=False); (output/'groups').mkdir()
    plan = dict(schema='expanded-comparison-plan-v1', profile=PROFILE.copy(), selection=selection,
                source_bindings=refs, frames=frames, reference_image_pair=list(reference),
                J='all original singleton observations; fixed cameras; DLT then XYZ-only LM',
                C='actual evaluate_six; independent certificate JSON readback; fixed18bit/common-y',
                old_xyz_used=False, foreground_certified=False, scale_status='unknown')
    bridge.write(output/'PLAN.json', plan)
    deadline = time.monotonic()+PROFILE['soft_seconds']; rows = []; paired = []; ledger = []
    def source_metadata(pid):
        observations = [heldout._observation(i, pid, images, points, cameras, poses)
                        for i in sorted(points[pid]['track'])]
        for obs in observations:
            obs['raw_binary64_hex'] = [float(v).hex() for v in obs['raw_pixels']]
        elements = []
        for image_id, index in sorted(points[pid]['track_elements']):
            xy = images[image_id]['observations'][index][:2]
            elements.append(dict(image_id=image_id, point2D_index=index, raw_pixels=list(xy),
                                 raw_binary64_hex=[float(v).hex() for v in xy]))
        return dict(observations=observations, original_track_elements=elements,
                    ambiguous_observations=points[pid]['ambiguous_observations'])
    for pair_index, selected in enumerate(selection['selected']):
        pair = selected['image_pair']; pc = [cameras[poses[i]['camera_id']] for i in pair]; pp = [poses[i] for i in pair]
        try:
            stereo = rect.calibrated_rectification(pc[0]['K'], pp[0]['R'], pp[0]['t'], pc[1]['K'], pp[1]['R'], pp[1]['t'])
            error = None
        except (ValueError, ArithmeticError) as exc: stereo = None; error = str(exc)
        bridge.write(output/f'pair{pair_index}_RECTIFICATION.json', dict(rectification=stereo, error=error))
        for group in selected['groups']:
            ids = group['point_ids']; raw = np.array([[v for i in pair for v in images[i]['observations'][points[pid]['track'][i]][:2]] for pid in ids])
            if not group['scheduled'] or time.monotonic() >= deadline:
                row = dict(status='NOT_EVALUATED_BUDGET', ablation=None)
            elif stereo is None: row = dict(status='PAIR_RECTIFICATION_REFUSED', reason=error, ablation=None)
            else: row = previous._case(raw, pc, pp, stereo, ids, 'positive', ids)
            row.update(pair_index=pair_index, image_pair=pair, point_ids=ids, raw_pixels=raw, block_index=group['block_index'])
            c = (row.get('ablation') or {}).get('C', {})
            c_ok = (c.get('status') == previous.WORLD_OK and c.get('exact_status') == 'CONSTRUCTED'
                    and c.get('independent', {}).get('status') == 'VERIFIED'
                    and c.get('projection_readback_exact') is True
                    and c.get('projection_metrics', {}).get('status') == 'NUMERIC_ROUNDTRIP_VERIFIED')
            samples = []
            evidence = f"groups/p{pair_index}_g{group['block_index']:04d}.json"
            for index, pid in enumerate(ids):
                metadata = source_metadata(pid); observations = metadata['observations']
                if time.monotonic() >= deadline or not group['scheduled']:
                    j = dict(status='NOT_EVALUATED_BUDGET', xyz=None, converged=False)
                else: j = fixed.fit(observations, max_iterations=PROFILE['max_iterations'])
                sample = dict(point_id=pid, C_xyz=c['world_xyz'][index] if c_ok else None,
                    J_xyz=j.get('xyz'), J_converged=j.get('converged', False), observations=observations,
                    C_status=c.get('status', row['status']), C_success=c_ok, J_details=j,
                    original_track_elements=metadata['original_track_elements'],
                    ambiguous_observations=points[pid]['ambiguous_observations'], evidence_file=evidence,
                    pair_index=pair_index, image_pair=pair, block_index=group['block_index'])
                sample['status'] = 'SAME_ID_PAIRED_NOT_ACCEPTED' if c_ok and j.get('xyz') is not None else 'UNPAIRED'
                samples.append(sample)
                if sample['status'] == 'SAME_ID_PAIRED_NOT_ACCEPTED': paired.append(sample)
                else: ledger.append(sample)
            row['samples'] = samples; bridge.write(output/evidence, row)
            rows.append(dict(evidence_file=evidence, status=row['status'], C_status=c.get('status'), point_ids=ids))
        for pid in selected['tail_point_ids']:
            ledger.append(dict(point_id=pid, status='UNPAIRED_SIX_GROUP_TAIL', image_pair=pair,
                               pair_index=pair_index, C_xyz=None, J_xyz=None, J_converged=False,
                               **source_metadata(pid)))
    for pid in selection['unassigned_point_ids']:
        ledger.append(dict(point_id=pid, status='UNPAIRED_PAIR_CAP_OR_INSUFFICIENT_UNCOVERED_TRACKS',
                           C_xyz=None, J_xyz=None, J_converged=False, **source_metadata(pid)))
    accounted = [r['point_id'] for r in paired+ledger]
    if sorted(accounted) != sorted(points) or len(set(accounted)) != len(accounted):
        raise ValueError('complete exactly-once track accounting failed')
    paired.sort(key=lambda r: r['point_id']); ledger.sort(key=lambda r: r['point_id'])
    bridge.write(output/'PAIRED_POINTS.json', dict(schema='expanded-same-id-pointset-v1', points=paired))
    bridge.write(output/'UNPAIRED_LEDGER.json', dict(points=ledger))
    for method in ('C', 'J') if paired else ():
        bridge.write_ply(output/f'{method}_same_ids.ply', [r['point_id'] for r in paired],
                         np.asarray([r[method+'_xyz'] for r in paired]).reshape(-1, 3))
    for ref in refs: bridge.verify_binding(ref)
    result = dict(schema='expanded-comparison-result-v1', status='CANDIDATES_GENERATED_NOT_ACCEPTED',
                  profile=PROFILE.copy(), source_unchanged=True, native_track_count=len(points),
                  selected_pair_count=len(selection['selected']), group_count=len(rows), groups=rows,
                  paired_count=len(paired), unpaired_count=len(ledger),
                  J_converged_count=sum(r['J_converged'] for r in paired),
                  unpaired_statuses=dict(Counter(r['status'] for r in ledger)),
                  points_file='PAIRED_POINTS.json', unpaired_file='UNPAIRED_LEDGER.json',
                  old_xyz_used=False, foreground_certified=False, quality_status='NOT_EVALUATED', scale_status='unknown')
    bridge.write(output/'RESULT.json', result)
    return result
