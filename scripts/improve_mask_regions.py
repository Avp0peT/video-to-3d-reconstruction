"""Automatically audit and supplement a completed region experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from integrations.video_masks.region_improvement import main

if __name__ == '__main__':
    raise SystemExit(main())
