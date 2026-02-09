"""Derive a PEP 440 version string from package metadata or Git."""

import os
import re


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
        desc = result.stdout.strip().lstrip("v")
        m = re.match(r"^(.+)-(\d+)-g([0-9a-f]+)$", desc)
        if not m:
            raise RuntimeError(f"Cannot parse git describe output: {desc!r}")
        ver, distance, sha = m.group(1), m.group(2), m.group(3)

        if int(distance) == 0:
            return ver
        return f"{ver}.dev{distance}+{sha}"

    except Exception:
        return "unknown"


__version__: str = _get_version()
