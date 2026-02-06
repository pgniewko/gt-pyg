"""Derive a PEP 440 version string from Git metadata."""

import os
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
        desc = result.stdout.strip().lstrip("v")
        version, distance, sha = desc.rsplit("-", 2)

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
