"""Derive a PEP 440 version string from Git or installed metadata."""

from ._version_utils import (
    _get_version,
    _get_version_from_git,
    _get_version_from_metadata,
    _normalize_prerelease,
)


__version__: str = _get_version()
