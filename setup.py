"""
Setup script for Reinforcement Pre-Training (RPT) package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
version = "0.2.0"

setup(
    name="reinforcement-pretraining",
    version=version,
    author="RPT Package Authors",
    author_email="",
    description="Reinforcement Pre-Training for Language Models - Implementation of ArXiv:2506.08007",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/reinforcement-pretraining",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
            "mypy",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
            "accelerate>=0.20.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
            "mypy",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "deepspeed>=0.9.0",
            "accelerate>=0.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rpt-train=rpt.cli:train_cli",
            "rpt-eval=rpt.cli:eval_cli",
        ],
    },
    package_data={
        "rpt": ["*.yaml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="reinforcement learning, language models, pre-training, AI, NLP, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/your-username/reinforcement-pretraining/issues",
        "Source": "https://github.com/your-username/reinforcement-pretraining",
        "Documentation": "https://reinforcement-pretraining.readthedocs.io/",
    },
)