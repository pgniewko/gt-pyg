"""Derive a PEP 440 version string from Git metadata."""

import os
import re
import subprocess


def _get_version() -> str:
    """Return a PEP 440-compliant version derived from ``git describe``.

    * **Tagged commit** → ``1.2.3``
    * **Untagged commit** → ``1.2.3.dev4+gabc1234``

    Falls back to ``importlib.metadata`` for installed packages that are
    no longer inside a Git checkout, and finally to ``"unknown"``.
    """
    try:
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
        # Use regex to anchor on the -N-gHASH suffix so that tags with
        # hyphens (e.g. v1.0.0-rc1) are parsed correctly.
        desc = result.stdout.strip().lstrip("v")
        m = re.match(r"^(.+)-(\d+)-g([0-9a-f]+)$", desc)
        if not m:
            raise RuntimeError(f"Cannot parse git describe output: {desc!r}")
        version, distance, sha = m.group(1), m.group(2), m.group(3)

        if int(distance) == 0:
            return version
        return f"{version}.dev{distance}+{sha}"

    except Exception:
        try:
            from importlib.metadata import version

            return version("gt_pyg")
        except Exception:
            return "unknown"


__version__: str = _get_version()
