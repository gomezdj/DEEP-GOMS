"""
DEEP-GOMS v2 — Package Setup
Makes `src` importable as the `deepgoms` package from anywhere in the repo.

Install in development mode (editable, no copy):
    pip install -e .

This means you can run scripts from the repo root without adding sys.path
hacks, and changes to src/ are reflected immediately.
"""

from setuptools import setup, find_packages

setup(
    name="deepgoms",
    version="2.0.0",
    description=(
        "DEEP-GOMS: Deep Evolutionary Ensemble Predictor for "
        "Gut OncoMicrobiome Signatures — Cancer Immunotherapy Biomarkers"
    ),
    author="gomezdj",
    url="https://github.com/gomezdj/DEEP-GOMS",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"deepgoms": "src"},
    install_requires=[
        "torch==2.5.1",
        "torchvision==0.20.1",
        "xgboost==2.1.4",
        "lightgbm==4.6.0",
        "scikit-learn==1.6.1",
        "scipy==1.15.2",
        "shap==0.46.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pyyaml==6.0.2",
        "anndata==0.10.9",
        "scanpy==1.10.4",
        "harmonypy==0.0.10",
        "pydeseq2==0.4.12",
        "torch-geometric==2.6.1",
        "networkx==3.4.2",
        "python-igraph==0.11.8",
        "leidenalg==0.10.2",
        "umap-learn==0.5.7",
        "celltypist==1.6.3",
        "matplotlib==3.9.4",
        "seaborn==0.13.2",
        "adjusttext==1.2.0",
        "tqdm==4.67.1",
    ],
    extras_require={
        "scvi": ["scvi-tools==1.2.2"],
        "gpu":  ["torch==2.5.1+cu124"],
    },
    entry_points={
        "console_scripts": [
            "deepgoms-preprocess=scripts.preprocess:main",
            "deepgoms-train=train_model:main",
            "deepgoms-lodo=lodo_cross_validation:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
