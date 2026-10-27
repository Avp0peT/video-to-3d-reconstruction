"""``python -m kozlov_lab.reconstruction`` — see :mod:`kozlov_lab.reconstruction.cli`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
