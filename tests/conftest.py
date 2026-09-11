"""Make the repository root importable regardless of the pytest working directory,
so `python -m pytest C:\\path\\to\\CCE-QOS\\tests` works from anywhere."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
