"""Read-only OpenMVS depth-map integration; no native runtime is required."""

from .dmap import (
    DMapError,
    DepthMap,
    HAS_CONF,
    HAS_DEPTH,
    HAS_NORMAL,
    HAS_VIEWS,
    KNOWN_FLAGS,
    MAX_BYTES,
    MAX_PIXELS,
    NO_ID,
    read_dmap,
)

__all__ = [
    "DMapError",
    "DepthMap",
    "HAS_CONF",
    "HAS_DEPTH",
    "HAS_NORMAL",
    "HAS_VIEWS",
    "KNOWN_FLAGS",
    "MAX_BYTES",
    "MAX_PIXELS",
    "NO_ID",
    "read_dmap",
]
