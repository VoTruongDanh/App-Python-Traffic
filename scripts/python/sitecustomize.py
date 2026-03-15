"""Ensure project root is importable when running utility scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
