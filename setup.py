from setuptools import setup, find_packages
import os

_version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_pyg", "_version.py")
_ns = {"__file__": _version_path}
with open(_version_path) as _f:
    exec(_f.read(), _ns)

setup(
    name="gt_pyg",
    version=_ns["__version__"],
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
        "examples": [
            "notebook",
            "ipykernel",
            "ipywidgets",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "scipy",
        ],
        "all": [
            "gt_pyg[dev]",
            "gt_pyg[examples]",
        ],
    },
    include_package_data=True,
)
