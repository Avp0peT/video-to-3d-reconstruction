"""Prepare/run/report a bounded mask-aware regional correspondence experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from integrations.video_masks.region_experiment import main

if __name__ == '__main__':
    raise SystemExit(main())
