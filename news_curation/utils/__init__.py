"""news_curation utils package.

Adds the parent of news_curation/ to sys.path so news scripts can import
from the project-wide `tools/` package.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
