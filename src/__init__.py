"""Project package bootstrap."""

from __future__ import annotations

import os
from pathlib import Path


_runtime_cache = Path("/tmp/demandsense-rx-cache")
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache))
