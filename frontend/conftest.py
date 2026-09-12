"""Keep the web service out of the root pytest run unless its extras are installed.

The repository runs ``pytest --doctest-modules`` from the root, which imports every
module it finds. ``frontend/server`` needs FastAPI (see
``frontend/server/requirements.txt``), which the package's own ``dev`` extra does not
install, so collect it only when FastAPI is importable.
"""

from __future__ import annotations

import importlib.util

collect_ignore = [] if importlib.util.find_spec("fastapi") else ["server"]
