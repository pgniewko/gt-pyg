"""Shared helpers for deriving a PEP 440 version string."""

from __future__ import annotations

import os
import re
import subprocess


def _normalize_prerelease(ver: str) -> str:
    """Convert common pre-release tag formats to PEP 440."""
    ver = re.sub(r"[-.]?alpha[.-]?", "a", ver)
    ver = re.sub(r"[-.]?beta[.-]?", "b", ver)
    ver = re.sub(r"[-.]?rc[.-]?", "rc", ver)
    return ver


def _get_version_from_git() -> str:
    """Return the git-derived version for a source checkout."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        ["git", "describe", "--tags", "--long"],
        capture_output=True,
        text=True,
        cwd=repo_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    desc = result.stdout.strip().lstrip("v")
    parts = desc.rsplit("-", 2)
    if len(parts) != 3 or not parts[2].startswith("g"):
        raise RuntimeError(f"Cannot parse git describe output: {desc!r}")

    ver, distance, sha = parts[0], parts[1], parts[2][1:]
    ver = _normalize_prerelease(ver)

    if int(distance) == 0:
        return ver
    return f"{ver}.dev{distance}+{sha}"


def _get_version_from_metadata() -> str:
    """Return the installed distribution version."""
    from importlib.metadata import version

    return version("gt_pyg")


def _get_version() -> str:
    """Resolve a version from git, then package metadata, then a fallback."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        try:
            return _get_version_from_git()
        except Exception:
            pass

    try:
        return _get_version_from_metadata()
    except Exception:
        return "0+unknown"
