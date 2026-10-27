# SAM 3 integration (isolated)

Everything SAM 3 lives here so the project's NumPy-2 environment stays untouched.

- `python/` — standalone CPython 3.12.13 (astral-sh/python-build-standalone), fetched by `uv`.
- `env/` — virtual environment used by `scripts/segment_sam3.py` (`pillow`, `huggingface_hub`, `pytest`; **torch is NOT installed** — see `VERSIONS.json`).
- `src/sam3-repo/` — official <https://github.com/facebookresearch/sam3> at the frozen commit in `VERSIONS.json`. Licence: SAM License (`src/sam3-repo/LICENSE`).
- `sam3_adapter/` — the adapter package. Imports **only** the standard library plus Pillow; torch/sam3 are imported lazily inside `Sam3Backend.load()`.

## Layout of responsibilities

| module | responsibility |
|---|---|
| `contract.py` | object-mask-v1 constants, exit codes, counts recomputation |
| `imagery.py` | strict mask PNG read/write/verify, digests, overlays |
| `sourceio.py` | photo-io-v1 run binding (identity strictly from `frames.jsonl`) |
| `backends.py` | `Sam3Backend` (production) and `FakeBackend` (interface tests only) |
| `resources.py` | frozen RAM/VRAM thresholds and *soft* monitoring |
| `runner.py` | orchestration, manifest, statuses, refusal-to-overwrite |
| `probe.py` | environment / CUDA / HF-access probes (`python -m sam3_adapter.probe`) |

Never install SAM 3 (or torch) into the project environment; run the CLI with
`integrations/sam3/env/Scripts/python.exe`. See `docs/SAM3_USAGE.md`.
