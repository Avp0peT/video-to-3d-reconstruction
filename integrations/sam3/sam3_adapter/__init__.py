"""SAM 3 adapter for the object-mask-v1 contract.

Importing this package must never pull in torch or sam3: the interface,
validation, deadline, worker and fake-backend layers stay usable inside the
project's NumPy 2.5 environment, while the real backend lives in its own
isolated interpreter.
"""

from __future__ import annotations

__all__ = [
    "backends",
    "binding",
    "contract",
    "deadline",
    "imagery",
    "probe",
    "resources",
    "runner",
    "sourceio",
    "worker",
]
