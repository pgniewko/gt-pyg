import subprocess

from setuptools import setup, find_packages


def _git_version() -> str:
    """Derive a PEP 440 version from ``git describe``."""
    try:
        desc = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--long"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .lstrip("v")
        )
        version, distance, sha = desc.rsplit("-", 2)
        if int(distance) == 0:
            return version
        return f"{version}.dev{distance}+{sha}"
    except Exception:
        return "0.0.0"


setup(
    name="gt_pyg",
    version=_git_version(),
    description="Implementation of the Graph Transformer architecture in Pytorch-geometric",
    packages=find_packages(exclude=["tests*", "*.tests", "*.tests.*"]),
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "numpy",
        "rdkit",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    include_package_data=True,
)
