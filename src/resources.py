"""Resolution of the application resource root directory.

FIPread must locate bundled resources (``logo.png`` and ``models/saved_models``)
in two different situations:

* Running from source, where resources sit next to the project root directory.
* Running from a PyInstaller ``onefile`` executable, where the bundle is
  unpacked into a temporary directory exposed as ``sys._MEIPASS``.

``APPLICATION_ROOT`` is the single resolved value used by the rest of the code,
so the frozen/source distinction is handled in exactly one place.
"""

from __future__ import annotations

import sys
from pathlib import Path


def application_root() -> Path:
    """Return the directory that contains the bundled application resources.

    Returns:
        ``sys._MEIPASS`` when running inside a PyInstaller onefile bundle,
        otherwise the project root (the parent of the ``src`` package).
    """

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


# Resolved once at import time; used by the UI builder and the SVM predictor.
APPLICATION_ROOT = application_root()
