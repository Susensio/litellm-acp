"""Single source of truth for the package version at runtime.

Falls back to a static string if the package metadata isn't available yet
(e.g. running straight from a source checkout without an editable install).
"""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("litellm-acp")
except Exception:  # noqa: BLE001 - PackageNotFoundError or importlib quirks
    __version__ = "0.1.0"
