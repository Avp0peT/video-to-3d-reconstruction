"""Segmentation backends.

Two backends exist, deliberately separated:

* :class:`Sam3Backend` -- the production backend. It resolves the official
  ``facebook/sam3`` checkpoints (gated), then lazily imports torch/sam3 out of
  the isolated checkout. It is the only backend whose results may be labelled
  ``sam3_real_inference``.
* :class:`FakeBackend` -- a spec-driven stand-in used **only** to exercise the
  object-mask-v1 interface (statuses, counts, refusal-to-overwrite, hash
  binding). Its runs are labelled ``fake_backend_interface_test`` and must never
  be reported as SAM3 evidence.

Nothing here imports torch at module import time.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from . import imagery
from .binding import effective_options_digest, file_binding, sha256_file
from .visual_prompt import (VisualPromptError, validate_visual_prompt, visual_prompt_sha256,
                            frame_box, normalized_box)
from .contract import (
    FAKE_BACKEND_NAME,
    SAM3_BACKEND_NAME,
    SAM3_INTERNAL_RESOLUTION,
    SAM3_MASK_BINARIZATION,
    SAM3_SCORE_SEMANTICS,
)

SAM3_REPO_ID = "facebook/sam3"
SAM3_CHECKPOINT_FILE = "sam3.pt"
SAM3_CONFIG_FILE = "config.json"
SAM3_HF_URL = "https://huggingface.co/facebook/sam3"
HF_AUTH_DOCS = "https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication"

#: the image builder's architecture is fixed by code, not by config.json
SAM3_IMAGE_BUILDER_CONSUMES_CONFIG_FILE = False


class BackendUnavailable(RuntimeError):
    """Backend cannot be constructed at all (missing deps, weights, code)."""


class AuthRequired(RuntimeError):
    """Weights are gated and the current credentials were rejected."""


class BackendError(RuntimeError):
    """Backend constructed but a call failed."""


class ResourceLimit(RuntimeError):
    """A pre-registered RAM / VRAM budget was exceeded or cannot be proven."""


class TensorContractError(BackendError):
    """Upstream processor output does not match the pinned N x 1 x H x W contract."""


class ScoreContractError(BackendError):
    """A candidate score is neither null nor a finite raw number."""


@dataclass
class Candidate:
    candidate_id: str
    flat: bytes
    width: int
    height: int
    score: float | None
    score_semantics: str | None


# --------------------------------------------------------------------------- #
# score contract
# --------------------------------------------------------------------------- #
def validate_score(value, where: str) -> float | None:
    """Accept ``None`` or a finite raw number; reject NaN / Inf / non-numbers.

    Nothing is silently coerced to zero and no candidate is dropped: a bad score
    raises so the caller reports BACKEND_ERROR for that frame.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreContractError(f"{where}: score must be null or a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ScoreContractError(f"{where}: score must be finite, got {number!r}")
    return number


# --------------------------------------------------------------------------- #
# upstream processor output contract
# --------------------------------------------------------------------------- #
def _to_cpu_like(tensor):
    """Best-effort move of a tensor-like handle to the CPU side.

    Nothing here guarantees a copy; it only normalises the handle so that both
    the nested-list path and the array path see the same object.
    """
    for attribute in ("detach", "cpu"):
        if hasattr(tensor, attribute):
            try:
                tensor = getattr(tensor, attribute)()
            except TypeError:  # pragma: no cover
                pass
    if hasattr(tensor, "to"):
        try:
            tensor = tensor.to("cpu")
        except TypeError:  # pragma: no cover
            pass
    return tensor


def _as_nested(tensor, where: str):
    """Detach/copy to CPU and convert to nested lists without importing numpy.

    This remains the deterministic compatibility path for list-backed
    stand-ins. Real array handles are served by :func:`_array_like`, which
    avoids materialising a Python float tree for a whole plane.
    """
    tensor = _to_cpu_like(tensor)
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    if hasattr(tensor, "numpy"):  # pragma: no cover - fallback only
        return tensor.numpy().tolist()
    raise TensorContractError(f"{where}: value is not a tensor-like object ({type(tensor).__name__})")


def _array_like(tensor):
    """Return a real ``ndarray`` handle for a tensor-like object, else ``None``.

    ``None`` is a normal answer, not an error: it means "this object cannot
    hand out an array, use the nested-list path". Objects that expose
    ``numpy()`` (CPU torch tensors) or ``__array__`` (ndarrays) take the fast
    path; list-backed stubs do not, so their behaviour is unchanged.

    NumPy is imported lazily and only once such an object is actually seen, so
    the CLI-help / no-backend path stays free of it. A bfloat16 tensor is
    widened to float32 first, which is lossless because bfloat16 is a
    truncated float32.
    """
    if tensor is None:
        return None
    tensor = _to_cpu_like(tensor)
    has_numpy = hasattr(tensor, "numpy")
    if not has_numpy and not hasattr(tensor, "__array__"):
        return None
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy ships with both envs
        return None
    if has_numpy:
        try:
            array = tensor.numpy()
        except TypeError:
            # Only BF16 has the explicitly supported, lossless widening
            # contract. An unrelated numpy() failure must not coerce values.
            if str(getattr(tensor, "dtype", "")) != "torch.bfloat16":
                return None
            widen = getattr(tensor, "float", None)
            if widen is None:
                return None
            try:
                array = widen().numpy()
            except Exception:  # pragma: no cover - defensive
                return None
        except Exception:  # pragma: no cover - defensive
            return None
    else:
        try:
            array = np.asarray(tensor)
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        array = np.asarray(array)
        # Object arrays can contain valid Python numbers/None, and uncommon
        # dtypes have their own tolist semantics. Preserve the old validator
        # for those inputs rather than changing which values it accepts.
        if array.dtype.kind not in ("b", "i", "u", "f"):
            return None
        if array.dtype.kind == "f" and array.dtype.itemsize > 8:
            return None
        return array
    except Exception:  # pragma: no cover - defensive
        return None


def _raise_if_non_finite_array(array, where: str) -> None:
    """Array-level counterpart of :func:`_raise_if_non_finite`.

    Only bool / integer / floating dtypes are contemplated; anything else is
    reported the same way the nested path reports a non-numeric leaf.
    """
    import numpy as np  # noqa: PLC0415

    kind = array.dtype.kind
    if kind == "b":
        # The nested validator rejects boolean leaves, but an empty list has
        # no leaves to reject. Keep both cases identical on the array path.
        if array.size:
            raise TensorContractError(f"{where}: non-numeric value bool")
        return
    if kind in ("i", "u"):
        return
    if kind == "f":
        if not bool(np.isfinite(array).all()):
            raise TensorContractError(f"{where}: non-finite value in logits")
        return
    raise TensorContractError(f"{where}: non-numeric value {array.dtype.name}")


def _validate_plane_shape(labels, height: int, width: int, where: str) -> None:
    if not isinstance(labels, list) or len(labels) != height:
        raise TensorContractError(
            f"{where}: mask height {len(labels) if isinstance(labels, list) else type(labels).__name__} "
            f"!= canonical {height}"
        )
    for row in labels:
        if not isinstance(row, list) or len(row) != width:
            raise TensorContractError(
                f"{where}: mask width {len(row) if isinstance(row, list) else type(row).__name__} "
                f"!= canonical {width}"
            )


def _plane_to_bytes(labels, where: str) -> bytes:
    """Turn a validated H x W label plane into flat 0/255 bytes.

    Booleans map directly; numeric labels must be finite and exactly 0 or 1.
    ``bool(...)`` is never applied blindly to a nested structure, which is what
    previously let the singleton channel axis pass as pixels.
    """
    flat = bytearray()
    for y, row in enumerate(labels):
        for x, value in enumerate(row):
            if isinstance(value, bool):
                flat.append(255 if value else 0)
                continue
            if not isinstance(value, (int, float)):
                raise TensorContractError(f"{where}: label at ({y},{x}) is {type(value).__name__}, not numeric")
            if not math.isfinite(float(value)):
                raise TensorContractError(f"{where}: non-finite label {value!r} at ({y},{x})")
            if float(value) not in (0.0, 1.0):
                raise TensorContractError(f"{where}: non-binary label {value!r} at ({y},{x})")
            flat.append(255 if float(value) else 0)
    return bytes(flat)


def _plane_to_bytes_from_array(plane, where: str) -> bytes:
    """Array-level counterpart of :func:`_plane_to_bytes`.

    Finiteness is checked **before** any integer cast: casting first would map
    NaN to 0 and silently launder a non-finite label into a background pixel.
    Exact 0/1 membership is then required, and only afterwards is the
    contiguous ``uint8`` buffer produced (0 -> 0, otherwise -> 255), which is
    the same row-major byte order the nested path appends.
    """
    import numpy as np  # noqa: PLC0415

    array = np.asarray(plane)
    if array.ndim != 2:
        raise TensorContractError(f"{where}: mask plane must be 2-D, got shape {array.shape}")
    kind = array.dtype.kind
    if kind == "f":
        if not bool(np.isfinite(array).all()):
            raise TensorContractError(f"{where}: non-finite label in mask plane")
    elif kind not in ("b", "i", "u"):
        raise TensorContractError(f"{where}: label dtype {array.dtype.name} is not numeric")
    if not bool(np.logical_or(array == 0, array == 1).all()):
        raise TensorContractError(f"{where}: non-binary label in mask plane")
    foreground = np.ascontiguousarray(array != 0, dtype=np.uint8)
    return (foreground * np.uint8(255)).tobytes()


def _raise_if_non_finite(node, where: str, depth: int) -> None:
    if depth > 6:  # pragma: no cover - guard against pathological nesting
        return
    if isinstance(node, list):
        for item in node:
            _raise_if_non_finite(item, where, depth + 1)
        return
    if node is None:
        return
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise TensorContractError(f"{where}: non-numeric value {type(node).__name__}")
    if not math.isfinite(float(node)):
        raise TensorContractError(f"{where}: non-finite value {node!r}")


def _candidates_from_array(mask_array, score_values, count: int, height: int, width: int) -> list[Candidate]:
    """Build candidates straight from an N x 1 x H x W array.

    Contract checks are the same ones the nested path performs, in the same
    order, on the same logical values; the difference is only that the plane
    never becomes a Python structure.
    """
    if mask_array.ndim != 4:
        raise TensorContractError(f"masks must be rank 4 (N,1,H,W), got shape {tuple(mask_array.shape)}")
    if mask_array.shape[0] != count:
        raise TensorContractError(
            f"masks first dimension {mask_array.shape[0]} != declared {count}"
        )
    if mask_array.shape[1] != 1:
        raise TensorContractError(
            f"masks channel dimension must be 1, got {mask_array.shape[1]} (shape {tuple(mask_array.shape)})"
        )
    if (mask_array.shape[2], mask_array.shape[3]) != (height, width):
        raise TensorContractError(
            f"masks spatial size {(mask_array.shape[2], mask_array.shape[3])} != canonical {(height, width)}"
        )
    candidates: list[Candidate] = []
    for index in range(count):
        where = f"candidate {index}"
        candidates.append(
            Candidate(
                candidate_id=f"c{index:02d}",
                flat=_plane_to_bytes_from_array(mask_array[index, 0], where),
                width=width,
                height=height,
                score=validate_score(score_values[index], where),
                score_semantics=SAM3_SCORE_SEMANTICS,
            )
        )
    return candidates


def candidates_from_processor_output(masks, scores, width: int, height: int,
                                     mask_logits=None) -> list[Candidate]:
    """Convert the pinned processor output into candidates.

    The pinned upstream code sets ``state["masks"] = out_masks > 0.5`` where
    ``out_masks`` is ``interpolate(out_masks.unsqueeze(1), (h, w), bilinear)``,
    i.e. a **N x 1 x H x W** boolean tensor already at canonical size. All four
    dimensions are validated and the singleton channel is removed explicitly.

    Two equivalent paths exist. A real array handle (CPU torch tensor, ndarray)
    is validated and converted at array level without ever building a Python
    float tree for a whole plane; a list-backed stand-in keeps the original
    element-by-element path unchanged.
    """
    if masks is None:
        raise TensorContractError("processor returned no masks")
    shape = tuple(int(dimension) for dimension in (getattr(masks, "shape", None) or ()))
    if len(shape) != 4:
        raise TensorContractError(f"masks must be rank 4 (N,1,H,W), got shape {shape}")
    count, channels, mask_height, mask_width = shape
    if channels != 1:
        raise TensorContractError(f"masks channel dimension must be 1, got {channels} (shape {shape})")
    if (mask_height, mask_width) != (height, width):
        raise TensorContractError(
            f"masks spatial size {(mask_height, mask_width)} != canonical {(height, width)}"
        )
    if mask_logits is not None:
        logits_array = _array_like(mask_logits)
        if logits_array is not None:
            _raise_if_non_finite_array(logits_array, "masks_logits")
        else:
            _raise_if_non_finite(_as_nested(mask_logits, "masks_logits"), "masks_logits", 0)

    mask_array = _array_like(masks)
    nested = None
    if mask_array is None:
        nested = _as_nested(masks, "masks")
        if not isinstance(nested, list) or len(nested) != count:
            raise TensorContractError(
                f"masks first dimension {len(nested) if isinstance(nested, list) else '?'} != declared {count}"
            )

    if scores is None:
        raise TensorContractError("processor returned no scores")
    score_values = _as_nested(scores, "scores")
    if isinstance(score_values, list) and score_values and isinstance(score_values[0], list):
        score_values = [row[0] for row in score_values]
    if not isinstance(score_values, list):
        raise TensorContractError(f"scores must be a 1-D sequence, got {type(score_values).__name__}")
    if len(score_values) != count:
        raise TensorContractError(f"scores length {len(score_values)} != candidate count {count}")

    if mask_array is not None:
        return _candidates_from_array(mask_array, score_values, count, height, width)

    candidates: list[Candidate] = []
    for index, per_candidate in enumerate(nested):
        where = f"candidate {index}"
        if not isinstance(per_candidate, list) or len(per_candidate) != 1:
            raise TensorContractError(
                f"{where}: expected exactly one channel, got "
                f"{len(per_candidate) if isinstance(per_candidate, list) else '?'}"
            )
        plane = per_candidate[0]
        _validate_plane_shape(plane, height, width, where)
        candidates.append(
            Candidate(
                candidate_id=f"c{index:02d}",
                flat=_plane_to_bytes(plane, where),
                width=width,
                height=height,
                score=validate_score(score_values[index], where),
                score_semantics=SAM3_SCORE_SEMANTICS,
            )
        )
    return candidates


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


# --------------------------------------------------------------------------- #
# fake backend (interface tests only)
# --------------------------------------------------------------------------- #
class FakeBackend:
    """Deterministic backend driven by a JSON spec (interface tests only)."""

    name = FAKE_BACKEND_NAME

    def __init__(self, spec_path: str | Path):
        self.spec_path = Path(spec_path)
        self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.repo_dir = self.spec_path.parent

    def describe(self) -> dict:
        return {
            "name": self.name,
            "code_commit": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "config_sha256": None,
            "python_version": sys.version.split()[0],
            "torch_version": None,
            "cuda_runtime": None,
            "device": "none",
        }

    def bindings(self) -> dict:
        return {"fake_backend": True, "spec_path": str(self.spec_path)}

    def load(self) -> None:
        raise_on_load = self.spec.get("load_error")
        if raise_on_load:
            message = self.spec.get("load_error_message", "fake backend load_error")
            error = {
                "BackendUnavailable": BackendUnavailable,
                "AuthRequired": AuthRequired,
                "BackendError": BackendError,
                "ResourceLimit": ResourceLimit,
            }.get(raise_on_load)
            if error is None:
                raise BackendUnavailable(f"unknown fake load_error {raise_on_load!r}")
            raise error(message)

    def close(self) -> None:
        return None

    def segment(
        self, source_image: Path, prompt: str, object_id: str, width: int, height: int, input_index: int
    ) -> list[Candidate]:
        entry = self.spec.get("frames", {}).get(str(input_index), None)
        if entry is None:
            return []
        if "error" in entry:
            error = {"BackendError": BackendError, "ResourceLimit": ResourceLimit}[entry["error"]]
            raise error(entry.get("message", "fake frame error"))
        candidates: list[Candidate] = []
        for position, item in enumerate(entry.get("candidates", [])):
            where = f"frame[{input_index}] candidate {position}"
            mask_path = (self.repo_dir / item["mask_path"]).resolve()
            flat, mask_w, mask_h = imagery.read_mask_l(mask_path)
            if (mask_w, mask_h) != (width, height):
                raise imagery.ImageError(
                    f"{where}: mask size {(mask_w, mask_h)} != canonical {(width, height)}"
                )
            candidates.append(
                Candidate(
                    candidate_id=f"c{position:02d}",
                    flat=flat,
                    width=mask_w,
                    height=mask_h,
                    score=validate_score(item.get("score"), where),
                    score_semantics=item.get("score_semantics", "fake_score_semantics"),
                )
            )
        return candidates


# --------------------------------------------------------------------------- #
# production backend
# --------------------------------------------------------------------------- #
class Sam3Backend:
    """Official SAM 3 image model (production backend)."""

    name = SAM3_BACKEND_NAME

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint_path: str | Path | None,
        checkpoint_sha256: str | None,
        device: str,
        confidence_threshold: float,
        resolution: int = SAM3_INTERNAL_RESOLUTION,
        allow_hf_download: bool = False,
        config_path: str | Path | None = None,
        config_sha256: str | None = None,
        hf_revision: str | None = None,
        hf_cache_dir: str | Path | None = None,
        visual_prompt: dict | None = None,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_sha256 = config_sha256
        self.hf_revision = hf_revision
        self.hf_cache_dir = Path(hf_cache_dir).resolve() if hf_cache_dir else None
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.allow_hf_download = allow_hf_download
        self.token_present: bool | None = None
        self.weights_source: str | None = None
        self.hf_files: list[dict] = []
        self.hf_calls: list[dict] = []
        self._probed = False
        self._model = None
        self._processor = None
        self._torch = None
        self.visual_prompt = None
        if visual_prompt is not None:
            if type(visual_prompt) is not dict or type(visual_prompt.get("frames")) is not list:
                raise VisualPromptError("invalid positive box frames")
            self.visual_prompt = validate_visual_prompt(visual_prompt,
                [r.get("input_index") if type(r) is dict else None for r in visual_prompt["frames"]], None)

    # ------------------------------------------------------------- bindings
    def effective_options(self) -> dict:
        bpe = self.repo_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        cuda_autocast = self.device.split(":", 1)[0] == "cuda"
        options = {
            "builder": "sam3.model_builder.build_sam3_image_model",
            "version": "sam3",
            "repo_dir": str(self.repo_dir),
            "bpe_path": str(bpe),
            "bpe_sha256": sha256_file(bpe) if bpe.is_file() else None,
            "device": self.device,
            "eval_mode": True,
            "load_from_HF": False,
            "enable_segmentation": True,
            "enable_inst_interactivity": False,
            "compile": False,
            "inference_mode": True,
            "autocast_enabled": cuda_autocast,
            "autocast_device_type": "cuda" if cuda_autocast else None,
            "autocast_dtype": "bfloat16" if cuda_autocast else None,
            "autocast_scope": "set_image+set_text_prompt",
            "processor_resolution": self.resolution,
            "processor_confidence_threshold": self.confidence_threshold,
            "mask_binarization": SAM3_MASK_BINARIZATION,
            "score_semantics": SAM3_SCORE_SEMANTICS,
            "image_builder_consumes_config_file": SAM3_IMAGE_BUILDER_CONSUMES_CONFIG_FILE,
        }
        if self.visual_prompt is not None:
            options.update(prompt_mode="positive_boxes", visual_prompt_sha256=visual_prompt_sha256(self.visual_prompt),
                           autocast_scope="set_image+add_geometric_prompt")
        return options

    def bindings(self) -> dict:
        options = self.effective_options()
        return {
            "python_version": sys.version.split()[0],
            "torch_version": getattr(self._torch, "__version__", None),
            "cuda_runtime": getattr(getattr(self._torch, "version", None), "cuda", None),
            "code": {
                "repo_dir": str(self.repo_dir),
                "commit": _git_commit(self.repo_dir),
                "commit_source": "git rev-parse HEAD of the on-disk pinned checkout",
            },
            "config": {
                "config_sha256": effective_options_digest(options),
                "config_source": "effective_construction_and_inference_options",
                "config_source_note": (
                    "the image builder does not read config.json; the hash binds constructor "
                    "options and the scoped inference_mode/CUDA BF16 autocast policy"
                ),
                "effective_options": options,
                "declared_config_file": file_binding(self.config_path, self.config_sha256),
                "declared_config_file_consumed": SAM3_IMAGE_BUILDER_CONSUMES_CONFIG_FILE,
            },
            "weights": {
                "source": self.weights_source,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_file": file_binding(self.checkpoint_path, self.checkpoint_sha256),
                "hf_repo_id": SAM3_REPO_ID,
                "hf_revision": self.hf_revision,
                "hf_revision_pinned": self.hf_revision is not None,
                "hf_cache_dir": str(self.hf_cache_dir) if self.hf_cache_dir else None,
                "hf_files": list(self.hf_files),
                "hf_download_calls": list(self.hf_calls),
                "token_present": self.token_present,
                "token_value_recorded": False,
            },
        }

    def describe(self) -> dict:
        bindings = self.bindings()
        return {
            "name": self.name,
            "code_commit": bindings["code"]["commit"],
            "checkpoint_path": bindings["weights"]["checkpoint_path"],
            "checkpoint_sha256": bindings["weights"]["checkpoint_sha256"],
            "config_sha256": bindings["config"]["config_sha256"],
            "python_version": bindings["python_version"],
            "torch_version": bindings["torch_version"],
            "cuda_runtime": bindings["cuda_runtime"],
            "device": self.device,
        }

    # --------------------------------------------------------------- weights
    def resolve_weights(self) -> None:
        """Bind code + weights + configuration BEFORE any heavy import.

        Runs at most once per backend instance, so the outer pre-flight and
        ``load()`` share one resolution instead of downloading twice.
        """
        if self._probed:
            return
        if not self.repo_dir.is_dir():
            raise BackendUnavailable(f"sam3 checkout not found: {self.repo_dir}")
        bpe = self.repo_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.is_file():
            raise BackendUnavailable(f"BPE vocabulary missing: {bpe}")

        if self.checkpoint_path is not None:
            self._resolve_local()
        elif self.allow_hf_download:
            self._resolve_hf()
        else:
            raise BackendUnavailable(
                "no --checkpoint given and --allow-hf-download not set; "
                f"weights for {SAM3_REPO_ID} are gated at {SAM3_HF_URL}"
            )
        self._probed = True

    def _resolve_local(self) -> None:
        if not self.checkpoint_path.is_file():
            raise BackendUnavailable(f"checkpoint not found: {self.checkpoint_path}")
        if not self.checkpoint_sha256:
            raise BackendUnavailable(
                "a local checkpoint was given without --checkpoint-sha256; refusing before the heavy "
                "load so the weights can never be recorded as unverified"
            )
        actual = sha256_file(self.checkpoint_path)
        if actual != self.checkpoint_sha256:
            raise BackendUnavailable(
                f"checkpoint sha256 mismatch: expected {self.checkpoint_sha256}, got {actual}"
            )
        if self.config_path is not None:
            if not self.config_path.is_file():
                raise BackendUnavailable(f"declared config not found: {self.config_path}")
            if not self.config_sha256:
                raise BackendUnavailable(
                    "a config file was declared without --config-sha256; refusing before the heavy load"
                )
            if sha256_file(self.config_path) != self.config_sha256:
                raise BackendUnavailable("declared config sha256 mismatch")
        self.weights_source = "local_checkpoint"

    def _resolve_hf(self) -> None:
        try:
            from huggingface_hub import get_token, hf_hub_download  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env dependent
            raise BackendUnavailable(f"huggingface_hub is not importable: {exc}") from exc
        try:
            self.token_present = bool(get_token())
        except Exception:  # pragma: no cover
            self.token_present = None
        cache_dir = str(self.hf_cache_dir) if self.hf_cache_dir else None
        for filename in (SAM3_CONFIG_FILE, SAM3_CHECKPOINT_FILE):
            call = {"repo_id": SAM3_REPO_ID, "filename": filename,
                    "revision": self.hf_revision, "cache_dir": cache_dir}
            self.hf_calls.append(dict(call))
            try:
                resolved = hf_hub_download(**call)
            except Exception as exc:  # noqa: BLE001 - huggingface_hub raises many types
                status = getattr(getattr(exc, "response", None), "status_code", None)
                text = f"{type(exc).__name__}: {exc}"
                if status in (401, 403) or "401" in text or "403" in text or "gated" in text.lower():
                    raise AuthRequired(
                        "Hugging Face rejected access to the gated SAM 3 weights "
                        f"({SAM3_REPO_ID} at revision {self.hf_revision!r}); "
                        f"token_present={self.token_present}. Request access at {SAM3_HF_URL} and "
                        f"authenticate ({HF_AUTH_DOCS})."
                    ) from exc
                raise BackendUnavailable(f"could not resolve {filename}: {text}") from exc
            self.hf_files.append({
                "filename": filename,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "bytes": Path(resolved).stat().st_size,
                "repo_id": SAM3_REPO_ID,
                "revision": self.hf_revision,
            })
            if filename == SAM3_CONFIG_FILE:
                self.config_path = Path(resolved)
            else:
                self.checkpoint_path = Path(resolved)
        if self.checkpoint_sha256 is None and self.checkpoint_path is not None:
            self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.weights_source = "huggingface_hub"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        self.resolve_weights()
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        try:
            import torch  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - env dependent
            raise BackendUnavailable(f"torch is not importable in this interpreter: {exc}") from exc
        self._torch = torch
        torch.set_num_threads(2)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
            from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env dependent
            raise BackendUnavailable(f"sam3 package is not importable from {self.repo_dir}: {exc}") from exc

        bpe_path = self.repo_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        try:
            self._model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                device=self.device,
                eval_mode=True,
                checkpoint_path=str(self.checkpoint_path),
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            self._processor = Sam3Processor(
                self._model,
                resolution=self.resolution,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as exc:  # pragma: no cover - weights/hardware dependent
            raise BackendError(f"SAM 3 model construction failed: {type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        self._model = None
        self._processor = None

    # --------------------------------------------------------------- forward
    def segment(
        self, source_image: Path, prompt: str | None, object_id: str, width: int, height: int, input_index: int
    ) -> list[Candidate]:
        if self._processor is None:
            raise BackendError("segment() called before load()")
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable(f"Pillow is not importable: {exc}") from exc

        torch = self._torch
        try:
            box = None
            if self.visual_prompt is not None:
                if prompt is not None:
                    raise VisualPromptError("positive_boxes cannot call a text prompt")
                row = frame_box(self.visual_prompt, input_index)
                if (width, height) != (row["width"], row["height"]) or sha256_file(source_image) != row["image_sha256"]:
                    raise VisualPromptError("positive box image dimensions or SHA256 drift")
                box = normalized_box(row)
            elif not isinstance(prompt, str):
                raise VisualPromptError("text-only backend requires a text prompt")
            with Image.open(source_image) as handle:
                handle.load()
                image = handle.convert("RGB")
            if box is not None and (image.size != (width, height) or sha256_file(source_image) != row["image_sha256"]):
                raise VisualPromptError("positive box canonical image changed while reading")
            # The pinned upstream image example wraps both image and text work
            # in CUDA BF16 autocast. Its fused vision path emits BF16 activations.
            # Use a scoped context so errors cannot leak a global precision mode.
            amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if self.device.split(":", 1)[0] == "cuda" else nullcontext())
            with torch.inference_mode(), amp:
                state = self._processor.set_image(image)
                if box is None:
                    state = self._processor.set_text_prompt(prompt=prompt, state=state)
                else:
                    state = self._processor.add_geometric_prompt(box=box, label=True, state=state)
            candidates = candidates_from_processor_output(
                state.get("masks"),
                state.get("scores"),
                width,
                height,
                mask_logits=state.get("masks_logits"),
            )
            del state
            if torch is not None and getattr(getattr(torch, "cuda", None), "empty_cache", None):
                torch.cuda.empty_cache()
            return candidates
        except (BackendError, BackendUnavailable, AuthRequired, ResourceLimit):
            raise
        except Exception as exc:
            out_of_memory = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
            if isinstance(out_of_memory, type) and isinstance(exc, out_of_memory):
                raise ResourceLimit(f"CUDA out of memory: {exc}") from exc
            raise BackendError(f"SAM 3 forward failed: {type(exc).__name__}: {exc}") from exc


def default_thread_policy() -> dict:
    """Environment variables this process sets for thread / bytecode discipline."""
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }


__all__ = [
    "AuthRequired",
    "BackendError",
    "BackendUnavailable",
    "Candidate",
    "FakeBackend",
    "ResourceLimit",
    "Sam3Backend",
    "ScoreContractError",
    "TensorContractError",
    "candidates_from_processor_output",
    "default_thread_policy",
    "validate_score",
]
