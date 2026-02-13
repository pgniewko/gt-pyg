"""Derive a PEP 440 version string from package metadata or Git."""

import os
import re


def _normalize_prerelease(ver: str) -> str:
    """Convert common pre-release tag formats to PEP 440.

    Examples:
        1.6.0-beta.1  → 1.6.0b1
        1.6.0-alpha.2 → 1.6.0a2
        1.6.0-rc.3    → 1.6.0rc3
        2.0.0a1       → 2.0.0a1   (already compliant, unchanged)
        1.5.8         → 1.5.8     (no pre-release, unchanged)
    """
    ver = re.sub(r"[-.]?alpha[.-]?", "a", ver)
    ver = re.sub(r"[-.]?beta[.-]?", "b", ver)
    ver = re.sub(r"[-.]?rc[.-]?", "rc", ver)
    return ver


def _get_version() -> str:
    """Return a PEP 440-compliant version string.

    Resolution order:

    1. ``importlib.metadata`` — fast, no subprocess; works for installed
       packages and in environments without git (Docker, CI tarballs).
    2. ``git describe --tags --long`` — used during local development in
       a git checkout where the package may not be installed.
    3. ``"unknown"`` — last resort.
    """
    # 1. Try installed package metadata first (no subprocess needed)
    try:
        from importlib.metadata import version

        return version("gt_pyg")
    except Exception:
        pass

    # 2. Fall back to git describe for development checkouts
    try:
        import subprocess

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "describe", "--tags", "--long"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        # git describe --tags --long  →  v1.2.3-N-ghash
        # Use rsplit to split from the right, so tags containing
        # hyphens (e.g. v1.6.0-beta.1) are handled correctly.
        desc = result.stdout.strip().lstrip("v")
        parts = desc.rsplit("-", 2)
        if len(parts) != 3 or not parts[2].startswith("g"):
            raise RuntimeError(f"Cannot parse git describe output: {desc!r}")

        ver, distance, sha = parts[0], parts[1], parts[2][1:]  # strip 'g' prefix
        ver = _normalize_prerelease(ver)

        if int(distance) == 0:
            return ver
        return f"{ver}.dev{distance}+{sha}"

    except Exception:
        return "unknown"


__version__: str = _get_version()
