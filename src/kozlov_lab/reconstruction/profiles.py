"""Versioned profile selector (``sfm-native-profile-v1``).

The frozen ``sfm-native-v1`` contract fixes per-image independent cameras.  This
module adds a *selectable* profile that changes exactly one thing — the extraction
camera mode — without touching the default, which must keep behaving exactly as it
did and must not acquire the new selector's fields in its recorded options.

Three profiles exist:

``t02a-v1`` (default)
    The frozen behaviour: ``contract_version = sfm-native-v1``, per-image
    independent cameras, ``CameraMode.PER_IMAGE``.  Its ``effective_options``
    deliberately stays free of any ``profile_id`` field, because the contract says
    adding a selector must not rewrite the default's recorded options.

``cad02a-shared-v1``
    ``contract_version = sfm-native-profile-v1``, ``profile_id`` recorded at the top
    level and in ``effective_options``, ``camera_sharing = "single"`` and
    ``CameraMode.SINGLE``.  A shared run must never be able to pass itself off as an
    independent-camera run, so the version string differs as well as the field.

Keeping this in its own module rather than editing constants in place is what makes
the change auditable: the old default remains reachable, and the old implementation
digest belongs to the retained snapshot rather than to the new files.

``cad-object-v1`` keeps shared intrinsics, 1024 pixels, two native threads and
4096 requested SIFT features, but accepts up to 96 distinct canonical views.
It requires the numeric runtime thread environment to be set to two before
Python starts. Its separate contract and fingerprint do not replace the closed
12-view experiment or change either legacy profile.

``cad-object-video-v1`` is the explicit whole-video-192-v1 SfM profile. It
retains the object profile numeric and runtime settings, with a distinct
contract and fingerprint and a strict capacity of 192 distinct canonical views.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_CAD02A_SHARED_V1",
    "PROFILE_CAD_OBJECT_V1",
    "PROFILE_CAD_OBJECT_PINHOLE_V1",
    "PROFILE_CAD_OBJECT_VIDEO_V1",
    "PROFILE_CONTRACT_VERSION",
    "PROFILE_T02A_V1",
    "VALID_PROFILES",
    "ProfileError",
    "camera_mode",
    "camera_sharing",
    "contract_version",
    "effective_profile_fields",
    "is_shared",
    "require_valid_profile",
    "summary",
]

PROFILE_T02A_V1 = "t02a-v1"
PROFILE_CAD02A_SHARED_V1 = "cad02a-shared-v1"
PROFILE_CAD_OBJECT_V1 = "cad-object-v1"
PROFILE_CAD_OBJECT_PINHOLE_V1 = "cad-object-pinhole-v1"
PROFILE_CAD_OBJECT_VIDEO_V1 = "cad-object-video-v1"
MAX_OBJECT_VIDEO_IMAGES = 192
MAX_OBJECT_IMAGES = 96
RUNTIME_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

DEFAULT_PROFILE = PROFILE_T02A_V1
VALID_PROFILES = (PROFILE_CAD_OBJECT_PINHOLE_V1, PROFILE_T02A_V1, PROFILE_CAD02A_SHARED_V1, PROFILE_CAD_OBJECT_V1, PROFILE_CAD_OBJECT_VIDEO_V1)

#: The contract version each profile reports.  The default keeps the frozen string.
PROFILE_CONTRACT_VERSION: dict[str, str] = {
    PROFILE_T02A_V1: "sfm-native-v1",
    PROFILE_CAD02A_SHARED_V1: "sfm-native-profile-v1",
    PROFILE_CAD_OBJECT_V1: "sfm-native-object-v1",
    PROFILE_CAD_OBJECT_PINHOLE_V1: "sfm-native-object-pinhole-v1",
    PROFILE_CAD_OBJECT_VIDEO_V1: "sfm-native-object-video-v1",
}

#: How many cameras the extractor is asked to produce.
_CAMERA_SHARING: dict[str, str] = {
    PROFILE_T02A_V1: "per_image_independent",
    PROFILE_CAD02A_SHARED_V1: "single",
    PROFILE_CAD_OBJECT_V1: "single",
    PROFILE_CAD_OBJECT_PINHOLE_V1: "single",
    PROFILE_CAD_OBJECT_VIDEO_V1: "single",
}

#: The name of the native ``CameraMode`` enum member the worker must use.  This is
#: the single behavioural difference between the two profiles.
_CAMERA_MODE: dict[str, str] = {
    PROFILE_T02A_V1: "PER_IMAGE",
    PROFILE_CAD02A_SHARED_V1: "SINGLE",
    PROFILE_CAD_OBJECT_V1: "SINGLE",
    PROFILE_CAD_OBJECT_PINHOLE_V1: "SINGLE",
    PROFILE_CAD_OBJECT_VIDEO_V1: "SINGLE",
}


class ProfileError(ValueError):
    """An unknown or unusable profile selector."""


def _reject(name: Any) -> None:
    """The single place the "not a valid selector" message is worded.

    Non-string selectors are refused explicitly rather than by dictionary lookup:
    ``1 in VALID_PROFILES`` is False but ``1 == "t02a-v1"`` comparisons look like
    ordinary membership tests, so an unhashable or non-string value must not be
    allowed to reach a ``KeyError`` or a misleading "unknown profile 1" that reads
    like a typo rather than a wrong type.
    """
    if not isinstance(name, str):
        raise ProfileError(
            f"profile selector must be a string, got {type(name).__name__} ({name!r}); "
            f"legal values are {', '.join(VALID_PROFILES)}"
        )
    raise ProfileError(
        f"unknown profile {name!r}; legal values are {', '.join(VALID_PROFILES)}"
    )


def require_valid_profile(name: str) -> str:
    """Return a validated profile name, or raise with the legal values listed."""
    if not isinstance(name, str) or name not in VALID_PROFILES:
        _reject(name)
    return name


def is_shared(profile: str) -> bool:
    return require_valid_profile(profile) in (PROFILE_CAD02A_SHARED_V1, PROFILE_CAD_OBJECT_V1, PROFILE_CAD_OBJECT_VIDEO_V1, PROFILE_CAD_OBJECT_PINHOLE_V1)


def contract_version(profile: str) -> str:
    return PROFILE_CONTRACT_VERSION[require_valid_profile(profile)]


def camera_sharing(profile: str) -> str:
    return _CAMERA_SHARING[require_valid_profile(profile)]


def camera_mode(profile: str) -> str:
    """The ``CameraMode`` member name the extraction stage must request."""
    return _CAMERA_MODE[require_valid_profile(profile)]


def effective_profile_fields(profile: str) -> dict[str, Any]:
    """Fields ``effective_options`` must gain for this profile.

    The default returns nothing: the contract requires that adding the selector does
    not alter the default profile's recorded options, so ``camera_sharing`` keeps its
    existing value and no ``profile_id`` appears.
    """
    require_valid_profile(profile)
    if profile == PROFILE_T02A_V1:
        return {}
    if profile in (PROFILE_CAD_OBJECT_V1, PROFILE_CAD_OBJECT_VIDEO_V1, PROFILE_CAD_OBJECT_PINHOLE_V1):
        return {"profile_id": profile, "camera_sharing": "single",
                "max_distinct_images": MAX_OBJECT_VIDEO_IMAGES if profile == PROFILE_CAD_OBJECT_VIDEO_V1 else MAX_OBJECT_IMAGES,
                "runtime_thread_environment": {key: "2" for key in RUNTIME_THREAD_ENV}}
    return {"profile_id": profile, "camera_sharing": _CAMERA_SHARING[profile]}


def summary(profile: str) -> dict[str, Any]:
    """A compact record of what the selector implies, for logs and handovers."""
    require_valid_profile(profile)
    return {
        "profile_id": profile,
        "contract_version": PROFILE_CONTRACT_VERSION[profile],
        "camera_sharing": _CAMERA_SHARING[profile],
        "camera_mode": _CAMERA_MODE[profile],
        "is_default": profile == DEFAULT_PROFILE,
    }
