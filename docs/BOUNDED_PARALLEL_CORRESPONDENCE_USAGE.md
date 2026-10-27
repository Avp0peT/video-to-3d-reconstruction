# Bounded theoretical parallel correspondence CLI

`scripts/search_parallel_correspondence.py` searches unknown label correspondence between two **complete** rational 2D views under the exact theoretical parallel-projection model. Each view has the same 4–7 distinct points. This is a finite mathematical branch of the 1996 construction, not a photo matcher, camera calibration, metric reconstruction, or CAD completion.

The two views have independent integer labels. Labels do not imply a correspondence, including when their numerical values happen to coincide. Input order is also irrelevant; the bounded algorithm enumerates explicit bijections. Coordinates use exact integers or canonical reduced rational objects. Labels are signed integers of at most 64 bits. Each coordinate's reduced numerator absolute value and denominator are at most `10^12`, matching the existing parallel-construction profile. Duplicate labels/coordinates are rejected; points are neither sampled nor merged.

```json
{
  "schema": "bounded-parallel-correspondence-input-v1",
  "model": "exact-parallel-theory",
  "source_xy": [
    {"label": 42, "xy": [0, 0]},
    {"label": -3, "xy": [1, 0]},
    {"label": 7, "xy": [0, 1]},
    {"label": 11, "xy": [1, 1]}
  ],
  "target_xy": [
    {"label": 55, "xy": [2, 3]},
    {"label": 108, "xy": [0, 0]},
    {"label": -10, "xy": [0, 1]},
    {"label": 77, "xy": [1, 0]}
  ]
}
```

A rational coordinate is `{"numerator":"1","denominator":"2"}`. Require canonical decimal strings, a positive denominator and a reduced fraction. Boolean/float coordinates and labels, repeated JSON keys, NaN/Infinity, inconsistent schema/model and unequal view cardinality are rejected. Input JSON is limited to 4 MiB.

```text
python -B -X utf8 scripts/search_parallel_correspondence.py --input views.json --output new-search.json
python -B -X utf8 scripts/search_parallel_correspondence.py --input views.json --verify new-search.json --output new-check.json
```

The script inserts the repository `src` path itself. Defaults are at most 5040 candidate correspondences, 20000000 work units, and 30 cooperative seconds; configurable with `--max-candidates`, `--max-work`, and `--seconds`. Construction and tagged-readback independent assessment share the remaining work/time. Verification-only mode uses a fresh explicitly requested bounded verification budget and never searches correspondences.

The CLI output is `bounded-parallel-correspondence-cli-result-v1`. It separately reports `search_status`, `search_complete`, `solutions_count`, `unsupported_count`, `independent_check`, and `total_work_used`. Every visited candidate retains its mapping and status in the ledger, including accepted, rank-rejected, unsupported planar, and unresolved branch candidates. All accepted records are preserved, without assuming one unique matching or deduplicating affine representatives. The result also binds the source input byte hash, resolved input path and exact CLI argument list. Verification mode binds the saved evidence byte hash too.

`SEARCH_COMPLETE` describes the bounded algorithm's declared complete ledger. `PARTIAL_UNKNOWN` retains any accepted results together with unvisited or unresolved scope. The independent assessment verifies the supplied claims and identities; `VERIFIED` does not replace `search_complete`. Empty accepted sets, partial searches, unsupported candidates and exhausted verification budgets are not complete reconstruction results. `physical_scale` remains unknown.

The output's `result` field uses `bounded-parallel-correspondence-transport-v1` and `exact-center-tags-v1`. The base document is tagged with existing `exact_center.encode_exact/decode_exact`. Each ledger candidate is carried in its own independently bounded tagged chunk with a consecutive offset; `accepted_indices` explicitly refer to accepted ledger records. `decode_result_transport` reconstructs the original ledger and accepted tuples and checks their identity binding. This preserves rational values, tuples and integer keys while avoiding the per-tree 20000-node decoder limit for an entire factorial-size ledger. Decoder limits are not relaxed. Saved CLI evidence is limited to 128 MiB when reading verification inputs.

Output files must be new. Both modes refuse overwrite and recheck the input byte hash before writing. Exit code 0 requires `SEARCH_COMPLETE`, declared `search_complete: true`, at least one accepted solution, zero unsupported candidates, and an independently `VERIFIED` tagged result. Exit 1 covers other mathematical/evidence statuses, including partial or empty results. Exit 2 means malformed input/evidence or filesystem errors. None of these statuses establish photo quality or metric/physical acceptance.

The CLI uses pure CPU exact mathematical code. It does not read real photos, import PIL, run models/GPU/native tools, install dependencies, launch a shell or start a GUI server. Four- and five-point rational 3D projection fixtures with shuffled independent labels are used for CLI tests. A separate final seven-point run at `outputs/delegation/GENERAL_PIPELINE/20261013_algorithm_core/parallel7/` completed all 5040 mappings in one CLI launch: two accepted, zero unsupported, the independently supplied truth mapping present, and the complete saved ledger independently VERIFIED. Its 8.03-second observation is specific to that fixture, not a performance guarantee for every seven-point input. It retains both solutions, without claiming unique physical correspondence.
