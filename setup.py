from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(
    name="neurips-2021-nethack-competition",
    version="0.1.0",
    description="Starter-kit repository for NeurIPS 2021 NLE competition",
    license="MIT License",
    packages=[
        "nethack_baselines",
    ],
    install_requires=[
        "aicrowd-api",
        "aicrowd-gym",
        "einops",
        "hydra-core",
        "hydra_colorlog",
        "numpy",
        "omegaconf",
        "torch",
        "ray[rllib]==1.3.0",
    ],
)
