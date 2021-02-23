"""Setup script for qreps."""
from setuptools import find_packages, setup

extras = {
    "test": [
        "pytest>=5.0,<5.1",
        "flake8>=3.7.8,<3.8",
        "pydocstyle==4.0.0",
        "black>=19.10b0",
        "isort>=5.0.0",
        "pytest_cov>=2.7,<3",
        "mypy==0.750",
        "pre-commit",
    ],
    "logging": ["tensorboard>=2.0,<3"],
    "experiments": ["lsf_runner==0.0.5", "pandas==0.25.0", "dotmap>=1.3.0,<1.4.0"],
}

setup(
    name="qreps",
    version="0.0.1",
    author="Sebastian Curi",
    author_email="sebascuri@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "rllib @ git+ssh://git@github.com/sebascuri/rllib@master#egg=rllib"
        "numpy>=1.14,<2",
        "scipy>=1.3.0,<1.4.0",
        "torch>=1.5.0,<1.7.0",
        "gym>=0.15.4",
        "atari_py>=0.2.6",
        "tqdm>=4.0.0,<5.0",
        "matplotlib>=3.1.0",
        "tensorboardX>=2.0,<3",
    ],
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
